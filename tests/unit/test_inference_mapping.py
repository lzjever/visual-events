from __future__ import annotations

from visual_events_server.inference.base import (
    COCO17_KEYPOINT_NAMES,
    bbox_area,
    clip_bbox,
)
from visual_events_server.inference.ultralytics_pose import result_to_pose_detections


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
