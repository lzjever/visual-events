from __future__ import annotations

import pytest

from visual_events_server.inference.base import (
    COCO17_KEYPOINT_NAMES,
    bbox_area,
    clip_bbox,
)
from visual_events_server.inference import ultralytics_pose
from visual_events_server.inference.ultralytics_pose import (
    UltralyticsPoseBackend,
    result_to_pose_detections,
)
from visual_events_server.protocol import FrameMessage


class FakeArray:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        return self.value


class FakeBoxes:
    def __init__(self, *, xyxy, conf, cls):
        self.xyxy = FakeArray(xyxy)
        self.conf = FakeArray(conf)
        self.cls = FakeArray(cls)


class FakeKeypoints:
    def __init__(self, *, xy, conf):
        self.xy = FakeArray(xy)
        self.conf = FakeArray(conf)


class FakeResult:
    def __init__(self, *, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class StepClock:
    def __init__(self, values):
        self.values = list(values)

    def __call__(self):
        return self.values.pop(0)


def test_result_to_pose_detections_filters_persons_clamps_bbox_and_maps_keypoints():
    keypoints = [
        [[10.0 + idx, 20.0 + idx] for idx in range(17)],
        [[30.0 + idx, 40.0 + idx] for idx in range(17)],
        [[50.0 + idx, 60.0 + idx] for idx in range(17)],
    ]
    keypoint_conf = [
        [0.9 for _ in range(17)],
        [0.8 for _ in range(17)],
        [0.7 for _ in range(17)],
    ]
    result = FakeResult(
        boxes=FakeBoxes(
            xyxy=[
                [-10.0, 5.0, 1300.0, 725.0],
                [20.0, 30.0, 60.0, 80.0],
                [100.0, 100.0, 150.0, 200.0],
            ],
            conf=[0.86, 0.95, 0.2],
            cls=[0, 2, 0],
        ),
        keypoints=FakeKeypoints(xy=keypoints, conf=keypoint_conf),
    )

    detections = result_to_pose_detections(
        result,
        image_width=1280,
        image_height=720,
        conf_threshold=0.5,
    )

    assert len(detections.persons) == 1
    person = detections.persons[0]
    assert person.bbox_xyxy == (0.0, 5.0, 1280.0, 720.0)
    assert person.bbox_area == 1280.0 * 715.0
    assert person.confidence == 0.86
    assert len(person.keypoints) == 17
    assert [keypoint.name for keypoint in person.keypoints] == COCO17_KEYPOINT_NAMES
    assert person.keypoints[0].name == "nose"
    assert person.keypoints[0].x == 10.0
    assert person.keypoints[0].y == 20.0
    assert person.keypoints[0].confidence == 0.9


def test_result_to_pose_detections_drops_low_confidence_and_invalid_bboxes():
    result = FakeResult(
        boxes=FakeBoxes(
            xyxy=[
                [10.0, 10.0, 10.0, 50.0],
                [30.0, 30.0, 80.0, 90.0],
                [100.0, 100.0, 120.0, 130.0],
            ],
            conf=[0.99, 0.49, 0.5],
            cls=[0, 0, 0],
        ),
        keypoints=FakeKeypoints(
            xy=[[[0.0, 0.0] for _ in range(17)] for _ in range(3)],
            conf=[[0.0 for _ in range(17)] for _ in range(3)],
        ),
    )

    detections = result_to_pose_detections(
        result,
        image_width=1280,
        image_height=720,
        conf_threshold=0.5,
    )

    assert [person.bbox_xyxy for person in detections.persons] == [
        (100.0, 100.0, 120.0, 130.0)
    ]


def test_result_to_pose_detections_handles_empty_result():
    detections = result_to_pose_detections(
        FakeResult(boxes=None, keypoints=None),
        image_width=1280,
        image_height=720,
        conf_threshold=0.5,
    )

    assert detections.persons == []


def test_bbox_helpers_clip_and_report_invalid_area():
    assert clip_bbox((-10.0, 20.0, 1300.0, 900.0), 1280, 720) == (
        0.0,
        20.0,
        1280.0,
        720.0,
    )
    assert bbox_area((0.0, 20.0, 1280.0, 720.0)) == 896000.0
    assert bbox_area((5.0, 5.0, 5.0, 10.0)) == 0.0


@pytest.mark.asyncio
async def test_ultralytics_backend_reports_decode_infer_postprocess_phase_metrics(
    tmp_path,
    monkeypatch,
):
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake model")
    clock = StepClock([10.0, 10.001, 10.001, 10.006, 10.006, 10.008])
    backend = UltralyticsPoseBackend(
        model_path=model_path,
        device=None,
        imgsz=320,
        conf=0.5,
        clock=clock,
    )
    result = FakeResult(
        boxes=FakeBoxes(
            xyxy=[[1.0, 2.0, 11.0, 22.0]],
            conf=[0.9],
            cls=[0],
        ),
        keypoints=FakeKeypoints(xy=[[]], conf=[[]]),
    )
    predict_calls = []

    class FakeModel:
        def predict(self, **kwargs):
            predict_calls.append(kwargs)
            return [result]

    monkeypatch.setattr(
        ultralytics_pose,
        "_decode_jpeg",
        lambda jpeg_bytes: ("decoded-image", 100, 80),
    )
    backend._model = FakeModel()

    detections = await backend.infer(
        FrameMessage(
            camera="front",
            frame_id=1,
            timestamp_ms=1000,
            width=100,
            height=80,
            jpeg_bytes=b"\xff\xd8fake\xff\xd9",
        )
    )

    assert len(detections.persons) == 1
    assert predict_calls[0]["source"] == "decoded-image"
    assert backend.consume_phase_metrics() == {
        "decode": pytest.approx(1.0),
        "infer": pytest.approx(5.0),
        "postprocess": pytest.approx(2.0),
    }
    assert backend.consume_phase_metrics() == {}
