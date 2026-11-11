"""Character tokenizer with interspersed blank tokens (id 0)."""


class Tokenizer:
    """Character tokenizer; id 0 is the pad/blank symbol '_'. Unknown characters are dropped."""

    def __init__(self, symbols):
        self.symbols = list(symbols)
        self.index = {s: i for i, s in enumerate(self.symbols)}

    def encode(self, text, add_blank=True, return_unknown=False):
        ids, unknown = [], []
        for c in text:
            i = self.index.get(c)
            if i is None or i == 0:
                unknown.append(c)
            else:
                ids.append(i)
        if add_blank:
            out = [0] * (2 * len(ids) + 1)
            out[1::2] = ids
            ids = out
        return (ids, unknown) if return_unknown else ids
