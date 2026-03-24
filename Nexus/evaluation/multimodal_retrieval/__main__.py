from transformers import HfArgumentParser

from .arguments import MultimodalRetrievalEvalArgs, MultimodalRetrievalEvalModelArgs
from .runner import MultimodalRetrievalEvalRunner


def main():
    parser = HfArgumentParser((MultimodalRetrievalEvalArgs, MultimodalRetrievalEvalModelArgs))
    eval_args, model_args = parser.parse_args_into_dataclasses()

    runner = MultimodalRetrievalEvalRunner(eval_args=eval_args, model_args=model_args)
    runner.run()


if __name__ == "__main__":
    main()
