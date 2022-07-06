from transformers import pipeline

APPROX_CHUNK_SIZE = 100000000000000

def load_text_from_file(file):
    out = []

    with open(file, "r") as infile:
        for line in infile:
            out.append(line.strip())

    return " ".join([elem for elem in out if elem != ""])

def process_text_to_sentences(text):
    text = text.replace(".", ".<eos>");
    text = text.replace("!", "!<eos>");
    text = text.replace("?", "?<eos>");

    return [elem.strip() for elem in text.split("<eos>") if elem != ""]

def chunk_sentences(sentences):
    chunks = []
    current_chunk = 0

    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(" ")) <= APPROX_CHUNK_SIZE:
                chunks[current_chunk] = chunks[current_chunk] + " " + sentence
            else:
                current_chunk += 1
                chunks.append(sentence)
        else:
            chunks.append(sentence)

    return chunks

def summarize_chunks(summarizer, chunks, min_length, max_length):
    res = summarizer(chunks, min_length=min_length, max_length=max_length, do_sample=False)

    return " ".join([elem["summary_text"] for elem in res])

def main():
    summarizer = pipeline("summarization", model="t5-small")

    text = load_text_from_file("data/TEXT.md")
    sentences = process_text_to_sentences(text)
    chunks = chunk_sentences(sentences)

    # print(len(chunks))

    summary = summarize_chunks(summarizer, chunks, 300, 400)
    print(summary)
    print(len(summary.split(" ")))

if __name__ == "__main__":
    main()