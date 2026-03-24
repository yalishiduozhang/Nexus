from transformers import HfArgumentParser

from .arguments import (
    MultimodalEmbedderDataArguments,
    MultimodalEmbedderModelArguments,
    MultimodalEmbedderTrainingArguments,
)
from .runner import MultimodalEmbedderRunner


def main():
    parser = HfArgumentParser(
        (
            MultimodalEmbedderModelArguments,
            MultimodalEmbedderDataArguments,
            MultimodalEmbedderTrainingArguments,
        )
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    runner = MultimodalEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    runner.run()


if __name__ == "__main__":
    main()
