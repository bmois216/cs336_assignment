import regex as re


GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_pairs(ids):
    """
    ids: Iterable[int] -> Iterable[tuple(int, int)]
    """
    pairs = set()
    for pair in zip(ids, ids[1:]):
        pairs.add(pair)

    return pairs


def update(ids, pair, new_id):
    """
    Args:
        ids: list[int]
        pair: tuple(int, int)
        new_ids: int

    Returns:
        new_ids: list[int]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        cur_pair = tuple(ids[i:i+2])
        if cur_pair == pair:
            new_ids.append(new_id)
            i += 1
        else:
            new_ids.append(ids[i])

        i += 1
    
    return new_ids


class BaseTokenizer:
    def __init__(self, vocab, merges, special_tokens):
        """
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str]
        """
        self.vocab = {}
        self.vocab['int_to_byte'] = vocab
        self.vocab['byte_to_int'] = {v:k for k,v in vocab.items()}

        self.merges = {}    # ((id1, id2), merged pair id)
        for a, b in merges:
            id_pair = (self.vocab['byte_to_int'][a], self.vocab['byte_to_int'][b])
            self.merges[id_pair] = self.vocab['byte_to_int'][a+b]

        self.special_tokens = {}    # dict(string, int)
        if special_tokens:
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for token in special_tokens:
                token_byte = token.encode("utf8")
                if token_byte not in self.vocab['byte_to_int']:
                    self.vocab['byte_to_int'][token_byte] = len(self.vocab['byte_to_int'])
                    self.vocab['int_to_byte'][len(self.vocab['int_to_byte'])] = token_byte
                    self.special_tokens[token] = len(self.vocab['int_to_byte'])
                else:
                    self.special_tokens[token] = self.vocab['byte_to_int'][token_byte]

    
    def from_files(cls, vocab_path, merges_path, special_tokens=None, **kwargs):
        pass


    def _encode_chunk(self, text):
        """
        text: string -> id: list[int]
        """
        if text in self.special_tokens:
            return [self.special_tokens[text]]
        else:
            text_chunks = re.findall(GPT2_PRETOKENIZER_PATTERN, text)   # list[str]

            result = []
            for chunk in text_chunks:
                text_bytes = chunk.encode("utf8")
                ids = [self.vocab['byte_to_int'][bytes([b])] for b in chunk.encode("utf8")] # base unmerge ids

                while len(ids) >= 2:
                    pairs = get_pairs(ids)
                    high_priority_pair = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))

                    if high_priority_pair not in self.merges:
                        break

                    new_id = self.merges[high_priority_pair]
                    ids = update(ids, high_priority_pair, new_id)

                result.extend(ids)

            return result


    def encode(self, text):
        """
        text: string -> id: list[int]
        """
        if self.special_tokens:
            special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            special_split_chunk = re.split(special_pattern, text)
        else:
            special_split_chunk = [text]

        ids = []
        for chunk in special_split_chunk:
            ids += self._encode_chunk(chunk)

        return ids
    

    def encode_iterable(self, texts):
        """
        texts: Iterable[string] -> id: list[int]
        """
        for text in texts:
            ids = self.encode(text)

            for id in ids:
                yield id

    
    def decode(self, ids):
        """
        ids: list[int] -> text: string
        """
        text_bytes = b''.join([self.vocab['int_to_byte'][id] for id in ids])

        return text_bytes.decode("utf8", errors="replace")

