from Nexus import MultimodalEmbedder


def main():
    model = MultimodalEmbedder(
        model_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
        processor_name_or_path="Qwen/Qwen2-VL-2B-Instruct",
        model_type="qwen2_vl",
        trust_remote_code=True,
        normalize_embeddings=True,
        query_max_length=512,
        passage_max_length=1024,
        batch_size=2,
    )

    queries = [
        {
            "text": "Find the matching sneaker",
            "images": ["examples/multimodal_retrieval/data/media/query_0001.jpg"],
        },
        {
            "text": "A red backpack with multiple pockets",
        },
    ]
    passages = [
        {
            "text": "White sneaker with orange sole",
            "images": ["examples/multimodal_retrieval/data/media/target_0001.jpg"],
        },
        {
            "text": "Red hiking backpack",
            "images": ["examples/multimodal_retrieval/data/media/target_0002.jpg"],
        },
    ]

    query_embeddings = model.encode_queries(queries, convert_to_numpy=False)
    passage_embeddings = model.encode_corpus(passages, convert_to_numpy=False)
    scores = query_embeddings @ passage_embeddings.T
    print(scores)


if __name__ == "__main__":
    main()

