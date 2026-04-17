from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmbias.config import ExperimentConfig, load_config
from llmbias.experiments.correction_runner import CorrectionRunner
from llmbias.experiments.dataset_runner import DatasetRunner
from llmbias.experiments.detection_runner import DetectionRunner
from llmbias.experiments.end_to_end_runner import EndToEndRunner
from llmbias.models.hf_local import HFLocalLLM
from llmbias.models.hf_pipeline import HFPipelineLLM
from llmbias.models.mock import MockLLM
from llmbias.pipelines.correction_pipeline import CorrectionPipeline
from llmbias.pipelines.detection_pipeline import DetectionPipeline
from llmbias.pipelines.end_to_end_pipeline import EndToEndBiasPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM bias detection and correction framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Run only research content one")
    detect_parser.add_argument("--prompt", required=True, help="Input prompt")
    detect_parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to experiment config",
    )

    correct_parser = subparsers.add_parser("correct", help="Run only research content two")
    correct_parser.add_argument("--response", required=True, help="Biased response text to rewrite")
    correct_parser.add_argument("--risk-score", type=float, default=0.5, help="Bias risk score")
    correct_parser.add_argument("--span-text", help="Target biased span text")
    correct_parser.add_argument("--confidence", type=float, default=0.8, help="Bootstrap confidence")
    correct_parser.add_argument("--prompt", default="", help="Optional source prompt for bookkeeping")
    correct_parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to experiment config",
    )

    run_parser = subparsers.add_parser("run", help="Run the full end-to-end pipeline")
    run_parser.add_argument("--prompt", required=True, help="Input prompt")
    run_parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to experiment config",
    )

    dataset_parser = subparsers.add_parser("run-dataset", help="Run the full pipeline on a local dataset")
    dataset_parser.add_argument("--dataset", choices=["bbq"], required=True, help="Dataset name")
    dataset_parser.add_argument("--dataset-path", required=True, help="Path to local dataset directory")
    dataset_parser.add_argument("--split", default="test", help="Dataset split name")
    dataset_parser.add_argument("--subset", default="", help="Optional BBQ subset name, e.g. Age_ambig")
    dataset_parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to run")
    dataset_parser.add_argument("--output", default="", help="Optional JSONL output path")
    dataset_parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to experiment config",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _safe_load_config(getattr(args, "config", "configs/default.yaml"))
    model = _build_model(config)

    if args.command == "detect":
        pipeline = DetectionPipeline(model=model, config=config.detection)
        runner = DetectionRunner(pipeline=pipeline)
        result = runner.run_single(args.prompt)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "correct":
        pipeline = CorrectionPipeline(config=config.correction)
        runner = CorrectionRunner(pipeline=pipeline)
        result = runner.run_single(
            response_text=args.response,
            risk_score=args.risk_score,
            span_text=args.span_text,
            confidence=args.confidence,
            prompt=args.prompt,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "run":
        pipeline = EndToEndBiasPipeline(model=model, config=config)
        runner = EndToEndRunner(pipeline=pipeline, config=config)
        result = runner.run_single(args.prompt)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "run-dataset":
        pipeline = EndToEndBiasPipeline(model=model, config=config)
        runner = EndToEndRunner(pipeline=pipeline, config=config)
        dataset_runner = DatasetRunner(runner=runner)
        results = dataset_runner.run_bbq(
            dataset_path=args.dataset_path,
            split=args.split,
            subset=args.subset,
            limit=args.limit,
            output_path=args.output or None,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))


def _safe_load_config(path: str) -> ExperimentConfig:
    try:
        return load_config(path)
    except FileNotFoundError:
        return ExperimentConfig()


def _build_model(config: ExperimentConfig):
    if config.model.provider == "mock":
        return MockLLM()
    if config.model.provider == "hf_pipeline":
        return HFPipelineLLM(
            model_id=config.model.model_id,
            model_name=config.model.model_name,
            device_map=config.model.device_map,
            torch_dtype=config.model.torch_dtype,
            max_new_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
        )
    if config.model.provider == "hf_local":
        model_path = Path(config.model.model_path)
        return HFLocalLLM(
            model_path=str(model_path),
            model_name=config.model.model_name,
            device_map=config.model.device_map,
            torch_dtype=config.model.torch_dtype,
            max_new_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
        )
    raise ValueError(f"Unsupported model provider: {config.model.provider}")


if __name__ == "__main__":
    main()
