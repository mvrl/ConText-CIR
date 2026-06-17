import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Dict
from nltk.tree import Tree


@dataclass
class Span(object):
    left: int
    right: int


@dataclass
class SubNP(object):
    text: str
    span: Span


@dataclass
class AllNPs(object):
    nps: List[str]
    spans: List[Span]
    lowest_nps: List[SubNP]


class Parser(nn.Module):
    def __init__(self, text_encoder=None, tokenizer=None) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            import stanza
            self._nlp = stanza.Pipeline(
                lang="en", processors="tokenize,pos,constituency", use_gpu=False,
                download_method=None, verbose=False,
            )
        return self._nlp

    def get_sub_nps(self, tree: Tree, left: int, right: int) -> List[SubNP]:
        if isinstance(tree, str) or len(tree.leaves()) == 1:
            return []

        sub_nps: List[SubNP] = []
        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[: len(n_subtree_leaves)]
        assert right - left == n_leaves

        if tree.label() == "NP" and n_leaves > 1:
            sub_nps.append(SubNP(
                text=" ".join(tree.leaves()),
                span=Span(left=int(left), right=int(right)),
            ))

        for i, subtree in enumerate(tree):
            sub_nps += self.get_sub_nps(
                subtree,
                left=left + offset[i],
                right=left + offset[i] + n_subtree_leaves[i],
            )
        return sub_nps

    def get_all_nps(self, tree: Tree, full_sent: Optional[str] = None) -> AllNPs:
        start, end = 0, len(tree.leaves())
        all_sub_nps = self.get_sub_nps(tree, left=start, right=end)

        # Lowest (most specific) NPs: no other NP is contained within them.
        lowest_nps = []
        for i in range(len(all_sub_nps)):
            span = all_sub_nps[i].span
            is_lowest = True
            for j in range(len(all_sub_nps)):
                if i != j:
                    span2 = all_sub_nps[j].span
                    if span2.left >= span.left and span2.right <= span.right:
                        is_lowest = False
                        break
            if is_lowest:
                lowest_nps.append(all_sub_nps[i])

        all_nps = [n.text for n in all_sub_nps]
        spans = [n.span for n in all_sub_nps]
        if full_sent and full_sent not in all_nps:
            all_nps = [full_sent] + all_nps
            spans = [Span(left=start, right=end)] + spans

        return AllNPs(nps=all_nps, spans=spans, lowest_nps=lowest_nps)

    def extract_noun_phrases(self, text: str, max_nps: int = 10) -> Dict:
        """Constituency-parse noun phrases (branch-level preferred over leaf-level)."""
        doc = self.nlp(text)
        nps, spans = [], []
        for sent in doc.sentences:
            try:
                tree = Tree.fromstring(str(sent.constituency))
                all_nps = self.get_all_nps(tree, text)
                # Prefer branch-level NPs over leaf-level (larger spans first).
                sorted_nps = sorted(
                    zip(all_nps.nps[1:], all_nps.spans[1:]),  # skip full sentence
                    key=lambda x: x[1].right - x[1].left,
                    reverse=True,
                )
                for np_text, span in sorted_nps[:max_nps]:
                    if np_text not in nps:
                        nps.append(np_text)
                        spans.append(span)
            except Exception as e:
                print(f"Error parsing sentence: {e}")
                continue
        return {"nps": nps[:max_nps], "spans": spans[:max_nps]}
