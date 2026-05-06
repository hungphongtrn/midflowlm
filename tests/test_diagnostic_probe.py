import json
import pytest
from src.diagnostic.probe import ProbeExample, ProbeSet, select_probes


@pytest.fixture
def sample_stress_test_data(tmp_path):
    mmlu_results = tmp_path / "mmlu_pro_results.json"
    mmlu_results.write_text(json.dumps([
        {
            "id": "mmlu_001",
            "question": "What is X?",
            "choices": ["A. ...", "B. ...", "C. ..."],
            "answer": "E",
            "student_answer_T1": "C",
            "student_answer_T8": "C",
            "teacher_answer": "E",
            "student_correct_T1": False,
            "teacher_correct": True,
        },
        {
            "id": "mmlu_002",
            "question": "What is Y?",
            "choices": ["A. ...", "B. ..."],
            "answer": "F",
            "student_answer_T1": "F",
            "teacher_answer": "F",
            "student_correct_T1": True,
            "teacher_correct": True,
        },
    ]))
    arc_results = tmp_path / "arc_easy_results.json"
    arc_results.write_text(json.dumps([
        {
            "id": "arc_001",
            "question": "What is Z?",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "student_answer_T1": "B",
            "teacher_answer": "A",
            "student_correct_T1": False,
            "teacher_correct": True,
        },
    ]))
    return str(mmlu_results), str(arc_results)


class TestProbeSelection:
    def test_select_mmlu_probes_E_J_labels(self, sample_stress_test_data):
        mmlu_path, arc_path = sample_stress_test_data
        probes = select_probes(mmlu_pro_path=mmlu_path, arc_easy_path=arc_path)
        
        mmlu_probes = [p for p in probes if p.benchmark == "mmlu_pro"]
        assert len(mmlu_probes) == 1
        assert mmlu_probes[0].target_label in "EFGHIJ"
        assert not mmlu_probes[0].student_correct

    def test_select_arc_probes_teacher_correct_student_wrong(self, sample_stress_test_data):
        mmlu_path, arc_path = sample_stress_test_data
        probes = select_probes(mmlu_pro_path=mmlu_path, arc_easy_path=arc_path)
        
        arc_probes = [p for p in probes if p.benchmark == "arc_easy"]
        assert len(arc_probes) == 1
        assert arc_probes[0].teacher_correct
        assert not arc_probes[0].student_correct

    def test_probe_set_minimum_size(self, sample_stress_test_data):
        mmlu_path, arc_path = sample_stress_test_data
        probes = select_probes(
            mmlu_pro_path=mmlu_path, arc_easy_path=arc_path,
            min_mmlu=1, min_arc=1
        )
        assert len(probes) >= 2

    def test_probe_serialization(self, sample_stress_test_data):
        mmlu_path, arc_path = sample_stress_test_data
        probes = select_probes(mmlu_pro_path=mmlu_path, arc_easy_path=arc_path)
        probe_set = ProbeSet(probes=probes, checkpoint_source="test", seed=42)
        
        data = probe_set.to_dict()
        assert "probes" in data
        assert "checkpoint_source" in data
        assert data["seed"] == 42
        
        reloaded = ProbeSet.from_dict(data)
        assert len(reloaded.probes) == len(probes)
        for original, recovered in zip(probes, reloaded.probes):
            assert recovered.id == original.id
            assert recovered.benchmark == original.benchmark
            assert recovered.question == original.question
            assert recovered.choices == original.choices
            assert recovered.target_label == original.target_label
            assert recovered.student_correct == original.student_correct
            assert recovered.teacher_correct == original.teacher_correct
            assert recovered.student_answer_T1 == original.student_answer_T1
            assert recovered.teacher_answer == original.teacher_answer
