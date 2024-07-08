"""
Write reference sentences to the bleu folder during training
    args: (
        en_test: List[str], list of source language sentences;
        vi_test: List[str], list of target language sentences;
        md: MosesDetokenizer;
    )
"""
def write_reference_to_bleu_folder(en_test, vi_test, md):
    with open("bleu/en_test.txt", "w") as f:
        for e in en_test:
            f.write(md.detokenize(e.split(' ')) + "\n\n")

    with open("bleu/vi_test.txt", "w") as f:
        for v in vi_test:
            f.write(md.detokenize(v.split(' ')) + "\n\n")