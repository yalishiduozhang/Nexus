from Nexus import MultimodalEmbedder


def main():
    model = MultimodalEmbedder(
        model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        processor_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct",
        pooling_method="last_token",
        normalize_embeddings=True,
        use_chat_template=True,
        use_fp16=False,
    )

    queries = [
        {
            "text": "Find the page that explains rotary position embeddings.",
            "image_path": "./examples/query_page.png",
        }
    ]
    corpus = [
        {"text": "RoPE rotates hidden dimensions by position-dependent angles."},
        {"text": "This page describes AdamW hyperparameters."},
    ]

    q_emb = model.encode_queries(queries)
    d_emb = model.encode_corpus(corpus)
    print("query shape:", q_emb.shape)
    print("doc shape:", d_emb.shape)
    print("scores:", q_emb @ d_emb.T)


if __name__ == "__main__":
    main()
