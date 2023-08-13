"""
defines the symbols permitted in text input to the model

we use this rather than string.printable to make it more modular
"""

_pad        = "_"
_eos        = "~"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? "


# export given vocabulary:
symbols = [_pad, _eos] + list(_characters)
