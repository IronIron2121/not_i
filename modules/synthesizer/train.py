from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import audio
from models.tacotron import Tacotron
from synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from utils import ValueWindow, data_parallel_workaround
from utils.plot import plot_spectrogram
from utils.symbols import symbols
from utils.text import sequence_to_text
from vocoder.display import *


def np_now(x: torch.Tensor): 
    return x.detach().cpu().numpy()

# get the current time
def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def train(run_id: str, syn_dir: Path, models_dir: Path, save_every: int,  backup_every: int, force_restart: bool,
          hparams):
    # create the provided directory for modelsif it doesn't already exist
    models_dir.mkdir(exist_ok=True)
    
    # create directory pathways for models, wavs, mel-plots, metadeta and...whatever plots are    
    model_dir = models_dir.joinpath(run_id) # contains model weights    
    plot_dir = model_dir.joinpath("matplots") # contains plots logging training
    wav_dir = model_dir.joinpath("wavs") # contains model output wavs (I guess?)
    melspec_out_dir = model_dir.joinpath("mel-spectrograms") # contains model output mel-spectrograms
    metadata_folder = model_dir.joinpath("metadata") # contains metadata (embeddings, etc...)
    
    # create directories for all of them based on assigned pathways
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    melspec_out_dir.mkdir(exist_ok=True)
    metadata_folder.mkdir(exist_ok=True)
    
    # paths to model (weights) and metadata
    model_path = model_dir / f"synthesizer.pt"
    metadata_path = syn_dir.joinpath("train.txt")
    
    # various logging details (the path of the weights, )
    print(f"Checkpoint path: {model_path}")
    print(f"Loading training data from: {metadata_path}")

    # custom class that stores a past window of values
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    
    # use cuda if user has a CUDA GPU, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
        # make sure that batch-size is divisible by GPUs (for parallel processing)
        for session in hparams.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError("`batch_size` must be evenly divisible by n_gpus!")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # initialise the model with provided hyperparameters
    print("\nInitialising Synthesiser...\n")
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=hparams.tts_dropout,
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    # initialise our adam optimiser over this model's parameters
    optimizer = optim.Adam(model.parameters())

    # if user has indicated they want a fresh start, or the provided weights filepath is empty, start a fresh set of weights
    if force_restart or not model_path.exists():
        print("\nStarting the training of Synthesiser from scratch\n")
        # save fresh weights at provided path
        model.save(model_path)

        # export our model vocabulary to the desired filepath in metadata folder
        char_embedding_path = metadata_folder.joinpath("CharacterEmbeddings.tsv")
        with open(char_embedding_path, "w", encoding="utf-8") as f:
            for symbol in symbols:
                if symbol == " ":
                    symbol = "\\s"  # For visual purposes, swap space with \s

                f.write(f"{symbol}\n")
    else:
        # otherwise, just load the weights the user provided
        print(f"\nLoading weights at {model_path}")
        model.load(model_path, optimizer)
        print(f"Synthesiser weights loaded from step {model.step}")

    # initialize the dataset for training 
    this_metadata = syn_dir.joinpath("train.txt")
    this_mel = syn_dir.joinpath("mels")
    this_embed = syn_dir.joinpath("embeds")
    dataset = SynthesizerDataset(this_metadata, this_mel, this_embed, hparams)
    
    """ 
    GABE: each "session" describes a given segment of the learning schedule
    hparams.py contains options to change the reduction factor learning rate, 
    step, and batch size for each of these
    """
    for i, session in enumerate(hparams.tts_schedule):
        # get the model's current step
        current_step = model.get_step()
        
        # get reduction factor, learning rate, step limit and batch size corresponding to this session    
        r, lr, max_step, batch_size = session
        
        
        training_steps = max_step - current_step

        # check if current step is past maximum for session
        if current_step >= max_step:
            # if we've already done all sessions, end training and save the model
            if i == len(hparams.tts_schedule) - 1:
                # We have completed training. Save the model and exit
                model.save(model_path, optimizer)
                break
            else:
                # go to the following session
                continue
            
        # update the model's reduction factor
        model.r = r
        
        # create a list of tuples logging training hyperparameter details
        simple_table([(f"Steps with r={r}", str(training_steps // 1000) + "k Steps"),
                      ("Batch Size", batch_size),
                      ("Learning Rate", lr),
                      ("Outputs/Step (r)", model.r)])
        
        # update adam optimiser's learning rates for parameters
        for p in optimizer.param_groups:
            p["lr"] = lr
            
        # set up a function that pads data so batches can be processed
        collate_fn = partial(collate_synthesizer, r=r, hparams=hparams)
        # load the data set, shuffle it, and use the provided collate function
        data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
        
        # set total iterations to match length of dataset
        total_iters = len(dataset)
        # calculate steps for each epoch
        steps_per_epoch = np.ceil(total_iters / batch_size).astype(np.int32)
        epochs = np.ceil(training_steps / steps_per_epoch).astype(np.int32)

        for epoch in range(1, epochs+1):
            for i, (texts, mels, embeds, idx) in enumerate(data_loader, 1):
                start_time = time.time()

                # Generate stop tokens for training
                stop = torch.ones(mels.shape[0], mels.shape[2])
                for j, k in enumerate(idx):
                    stop[j, :int(dataset.metadata[k][4])-1] = 0

                texts = texts.to(device)
                mels = mels.to(device)
                embeds = embeds.to(device)
                stop = stop.to(device)

                # Forward pass
                # Parallelize model onto GPUS using workaround due to python bug
                if device.type == "cuda" and torch.cuda.device_count() > 1:
                    m1_hat, m2_hat, attention, stop_pred = data_parallel_workaround(model, texts, mels, embeds)
                else:
                    m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)

                # Backward pass
                m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                m2_loss = F.mse_loss(m2_hat, mels)
                stop_loss = F.binary_cross_entropy(stop_pred, stop)

                loss = m1_loss + m2_loss + stop_loss

                optimizer.zero_grad()
                loss.backward()

                if hparams.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.tts_clip_grad_norm)
                    if np.isnan(grad_norm.cpu()):
                        print("grad_norm was NaN!")

                optimizer.step()

                time_window.append(time.time() - start_time)
                loss_window.append(loss.item())

                step = model.get_step()
                k = step // 1000

                msg = f"| Epoch: {epoch}/{epochs} ({i}/{steps_per_epoch}) | Loss: {loss_window.average:#.4} | " \
                      f"{1./time_window.average:#.2} steps/s | Step: {k}k | "
                stream(msg)

                # Backup or save model as appropriate
                if backup_every != 0 and step % backup_every == 0 :
                    backup_fpath = model_path.parent / f"synthesizer_{k:06d}.pt"
                    model.save(backup_fpath, optimizer)

                if save_every != 0 and step % save_every == 0 :
                    # Must save latest optimizer state to ensure that resuming training
                    # doesn't produce artifacts
                    model.save(model_path, optimizer)

                # Evaluate model to generate samples
                epoch_eval = hparams.tts_eval_interval == -1 and i == steps_per_epoch  # If epoch is done
                step_eval = hparams.tts_eval_interval > 0 and step % hparams.tts_eval_interval == 0  # Every N steps
                if epoch_eval or step_eval:
                    for sample_idx in range(hparams.tts_eval_num_samples):
                        # At most, generate samples equal to number in the batch
                        if sample_idx + 1 <= len(texts):
                            # Remove padding from mels using frame length in metadata
                            mel_length = int(dataset.metadata[idx[sample_idx]][4])
                            mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
                            target_spectrogram = np_now(mels[sample_idx]).T[:mel_length]
                            attention_len = mel_length // model.r

                            eval_model(attention=np_now(attention[sample_idx][:, :attention_len]),
                                       mel_prediction=mel_prediction,
                                       target_spectrogram=target_spectrogram,
                                       input_seq=np_now(texts[sample_idx]),
                                       step=step,
                                       plot_dir=plot_dir,
                                       mel_output_dir=melspec_out_dir,
                                       wav_dir=wav_dir,
                                       sample_num=sample_idx + 1,
                                       loss=loss,
                                       hparams=hparams)

                # Break out of loop to update training schedule
                if step >= max_step:
                    break

            # Add line break after every epoch
            print("")


def eval_model(attention, mel_prediction, target_spectrogram, input_seq, step,
               plot_dir, mel_output_dir, wav_dir, sample_num, loss, hparams):
    # Save some results for evaluation
    attention_path = str(plot_dir.joinpath(f"attention_step_{step}_sample_{sample_num}"))
    save_attention(attention, attention_path)

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", time_string(), step, loss)
    plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram,
                     max_len=target_spectrogram.size // hparams.num_mels)
    print("Input at step {}: {}".format(step, sequence_to_text(input_seq)))
