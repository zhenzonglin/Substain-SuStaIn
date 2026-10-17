"""诊断应真实反映缺失输入；源目录保护须早于任何执行记录写入。"""
from pathlib import Path
import pytest
import yaml
from substain_sustain.config import load_config, validate_config
from substain_sustain.cli import doctor, main

ROOT=Path(__file__).resolve().parents[1]


def test_doctor_empty_sources_not_ready():
    cfg=load_config(ROOT/"config/analysis.yaml")
    cfg["cohorts"]={name:{} for name in ("LS","C3","GB")}
    result=doctor(cfg)
    assert result["cohort_paths_ready"] is False
    assert result["workflow_ready"] is False


def test_boolean_rejected_as_scientific_integer():
    cfg=load_config(ROOT/"config/analysis.yaml")
    cfg["model"]["k_max"]=True
    with pytest.raises(ValueError,match="正整数"):
        validate_config(cfg)


def test_cli_refuses_source_output_before_writing(tmp_path):
    cfg=load_config(ROOT/"config/analysis.yaml")
    source=tmp_path/"original"
    source.mkdir()
    cfg["cohorts"]["LS"]["derivatives"]=str(source)
    cfg["output_dir"]=str(source/"unsafe_output")
    config=tmp_path/"config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        main(["ingest","--config",str(config)])
    assert not (source/"unsafe_output").exists()
