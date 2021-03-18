#!/usr/bin/env python
"""Convert between COCO and PoseTrack2017 format."""
# pylint: disable=too-many-branches, too-many-locals, bad-continuation, unsubscriptable-object
from __future__ import print_function

import json
import logging
import os
import os.path as path

import click
import numpy as np
import tqdm

# from posetrack18_id2fname import posetrack18_fname2id, posetrack18_id2fname
from .posetrack18_id2fname import posetrack18_fname2id, posetrack18_id2fname

LOGGER = logging.getLogger(__name__)
POSETRACK18_LM_NAMES_COCO_ORDER = [
    "nose",
    "head_bottom",  # "left_eye",
    "head_top",  # "right_eye",
    "left_ear",  # will be left zeroed out
    "right_ear",  # will be left zeroed out
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
POSETRACK18_LM_NAMES = [  # This is used to identify the IDs.
    "right_ankle",
    "right_knee",
    "right_hip",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "head_bottom",
    "nose",
    "head_top",
]

SCORE_WARNING_EMITTED = False


def json_default(val):
    """Serialization workaround
    https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python."""
    if isinstance(val, np.int64):
        return int(val)
    raise TypeError


class Video:

    """
    A PoseTrack sequence.

    Parameters
    ==========

    video_id: str.
      A five or six digit number, potentially with leading zeros, identifying the
      PoseTrack video.
    """

    def __init__(self, video_id):
        self.posetrack_video_id = video_id  # str.
        self.frames = []  # list of Image objects.

    def to_new(self):
        """Return a dictionary representation for the PoseTrack18 format."""
        result = {"images": [], "annotations": []}
        for image in self.frames:
            image_json = image.to_new()
            image_json["vid_id"] = self.posetrack_video_id
            image_json["nframes"] = len(self.frames)
            image_json["id"] = int(image.frame_id)
            result["images"].append(image_json)
            for person_idx, person in enumerate(image.people):
                person_json = person.to_new()
                person_json["image_id"] = int(image.frame_id)
                person_json["id"] = int(image.frame_id) * 100 + person_idx
                result["annotations"].append(person_json)
        # Write the 'categories' field.
        result["categories"] = [
            {
                "supercategory": "person",
                "name": "person",
                "skeleton": [
                    [16, 14],
                    [14, 12],
                    [17, 15],
                    [15, 13],
                    [12, 13],
                    [6, 12],
                    [7, 13],
                    [6, 7],
                    [6, 8],
                    [7, 9],
                    [8, 10],
                    [9, 11],
                    [2, 3],
                    [1, 2],
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                    [5, 7],
                ],
                "keypoints": POSETRACK18_LM_NAMES_COCO_ORDER,
                "id": 1,
            }
        ]
        return result

    def to_old(self):
        """Return a dictionary representation for the PoseTrack17 format."""
        res = {"annolist": []}
        for image in self.frames:
            elem = {}
            im_rep, ir_list, imgnum = image.to_old()
            elem["image"] = [im_rep]
            elem["imgnum"] = [imgnum]
            if ir_list:
                elem["ignore_regions"] = ir_list
            elem["annorect"] = []
            for person in image.people:
                elem["annorect"].append(person.to_old())
            if image.people:
                elem['is_labeled'] = [1]
            else:
                elem['is_labeled'] = [0]
            res["annolist"].append(elem)
        return res

    @classmethod
    def from_old(cls, track_data):
        """Parse a dictionary representation from the PoseTrack17 format."""
        assert "annolist" in track_data.keys(), "Wrong format!"
        video = None
        for image_info in track_data["annolist"]:
            image = Image.from_old(image_info)
            if not video:
                video = Video(
                    path.basename(path.dirname(image.posetrack_filename)).split("_")[0]
                )
            else:
                assert (
                    video.posetrack_video_id
                    == path.basename(path.dirname(image.posetrack_filename)).split("_")[
                        0
                    ]
                )
            video.frames.append(image)
        return [video]

    @classmethod
    def from_new(cls, track_data):
        """Parse a dictionary representation from the PoseTrack17 format."""
        image_id_to_can_info = {}
        video_id_to_video = {}
        assert len(track_data["categories"]) == 1
        assert track_data["categories"][0]["name"] == "person"
        assert len(track_data["categories"][0]["keypoints"]) in [15, 17]
        conversion_table = []
        for lm_name in track_data["categories"][0]["keypoints"]:
            if lm_name not in POSETRACK18_LM_NAMES:
                conversion_table.append(None)
            else:
                conversion_table.append(POSETRACK18_LM_NAMES.index(lm_name))
        for lm_idx, lm_name in enumerate(POSETRACK18_LM_NAMES):
            assert lm_idx in conversion_table, "Landmark `%s` not found." % (lm_name)
        videos = []
        for image_id in [image["id"] for image in track_data["images"]]:
            image = Image.from_new(track_data, image_id)
            video_id = path.basename(path.dirname(image.posetrack_filename)).split(
                "_"
            )[0]
            if video_id in video_id_to_video.keys():
                video = video_id_to_video[video_id]
            else:
                video = Video(video_id)
                video_id_to_video[video_id] = video
                videos.append(video)
            video.frames.append(image)
            for person_info in track_data["annotations"]:
                if person_info["image_id"] != image_id:
                    continue
                image.people.append(Person.from_new(person_info, conversion_table))
        return videos


class Person:

    """
    A PoseTrack annotated person.

    Parameters
    ==========

    track_id: int
      Unique integer representing a person track.
    """

    def __init__(self, track_id):
        self.track_id = track_id
        self.landmarks = None  # None or list of dicts with 'score', 'x', 'y', 'id'.
        self.rect_head = None  # None or dict with 'x1', 'x2', 'y1' and 'y2'.
        self.rect = None  # None or dict with 'x1', 'x2', 'y1' and 'y2'.
        self.score = None  # None or float.

    def to_new(self):
        """
        Return a dictionary representation for the PoseTrack18 format.

        The fields 'image_id' and 'id' must be added to the result.
        """
        keypoints = []
        scores = []
        write_scores = (
            len([1 for lm_info in self.landmarks if "score" in lm_info.keys()]) > 0
        )
        for landmark_name in POSETRACK18_LM_NAMES_COCO_ORDER:
            try:
                try:
                    lm_id = POSETRACK18_LM_NAMES.index(landmark_name)
                except ValueError:
                    lm_id = -1
                landmark_info = [lm for lm in self.landmarks if lm["id"] == lm_id][0]
            except IndexError:
                landmark_info = {"x": 0, "y": 0, "is_visible": 0}
            is_visible = 1
            if "is_visible" in landmark_info.keys():
                is_visible = landmark_info["is_visible"]
            keypoints.extend([landmark_info["x"], landmark_info["y"], is_visible])
            if "score" in landmark_info.keys():
                scores.append(landmark_info["score"])
            elif write_scores:
                LOGGER.warning("Landmark with missing score info detected. Using 0.")
                scores.append(0.)
        ret = {
            "track_id": self.track_id,
            "category_id": 1,
            "keypoints": keypoints,
            "scores": scores,
            # image_id and id added later.
        }
        if self.rect:
            ret["bbox"] = [
                self.rect["x1"],
                self.rect["y1"],
                self.rect["x2"] - self.rect["x1"],
                self.rect["y2"] - self.rect["y1"],
            ]
        if self.rect_head:
            ret["bbox_head"] = [
                self.rect_head["x1"],
                self.rect_head["y1"],
                self.rect_head["x2"] - self.rect_head["x1"],
                self.rect_head["y2"] - self.rect_head["y1"],
            ]
        return ret

    def to_old(self):
        """Return a dictionary representation for the PoseTrack17 format."""
        keypoints = []
        for landmark_info in self.landmarks:
            if (
                landmark_info["x"] == 0
                and landmark_info["y"] == 0
                and "is_visible" in landmark_info.keys()
                and landmark_info["is_visible"] == 0
            ):
                # The points in new format are stored like this if they're unannotated.
                # Skip in that case.
                continue
            point = {
                "id": [landmark_info["id"]],
                "x": [landmark_info["x"]],
                "y": [landmark_info["y"]],
            }
            if "score" in landmark_info.keys():
                point["score"] = [landmark_info["score"]]
            if "is_visible" in landmark_info.keys():
                point["is_visible"] = [landmark_info["is_visible"]]
            keypoints.append(point)
        # ret = {"track_id": [self.track_id], "annopoints": keypoints}
        ret = {"track_id": [self.track_id], "annopoints": [{'point': keypoints}]}
        if self.rect_head:
            ret["x1"] = [self.rect_head["x1"]]
            ret["x2"] = [self.rect_head["x2"]]
            ret["y1"] = [self.rect_head["y1"]]
            ret["y2"] = [self.rect_head["y2"]]
        if self.score:
            ret["score"] = [self.score]
        return ret

    @classmethod
    def from_old(cls, person_info):
        """Parse a dictionary representation from the PoseTrack17 format."""
        global SCORE_WARNING_EMITTED  # pylint: disable=global-statement
        person = Person(person_info["track_id"][0])
        assert len(person_info["track_id"]) == 1, "Invalid format!"
        rect_head = {}
        rect_head["x1"] = person_info["x1"][0]
        assert len(person_info["x1"]) == 1, "Invalid format!"
        rect_head["x2"] = person_info["x2"][0]
        assert len(person_info["x2"]) == 1, "Invalid format!"
        rect_head["y1"] = person_info["y1"][0]
        assert len(person_info["y1"]) == 1, "Invalid format!"
        rect_head["y2"] = person_info["y2"][0]
        assert len(person_info["y2"]) == 1, "Invalid format!"
        person.rect_head = rect_head
        try:
            person.score = person_info["score"][0]
            assert len(person_info["score"]) == 1, "Invalid format!"
        except KeyError:
            pass
        person.landmarks = []
        if "annopoints" not in person_info.keys() or not person_info["annopoints"]:
            return person
        lm_x_values = []
        lm_y_values = []
        for landmark_info in person_info["annopoints"][0]["point"]:
            lm_dict = {
                "y": landmark_info["y"][0],
                "x": landmark_info["x"][0],
                "id": landmark_info["id"][0],
            }
            lm_x_values.append(lm_dict["x"])
            lm_y_values.append(lm_dict["y"])
            if "score" in landmark_info.keys():
                lm_dict["score"] = landmark_info["score"][0]
                assert len(landmark_info["score"]) == 1, "Invalid format!"
            elif not SCORE_WARNING_EMITTED:
                LOGGER.warning("No landmark scoring information found!")
                LOGGER.warning("This will not be a valid submission file!")
                SCORE_WARNING_EMITTED = True
            if "is_visible" in landmark_info.keys():
                lm_dict["is_visible"] = landmark_info["is_visible"][0]
            person.landmarks.append(lm_dict)
            assert (
                len(landmark_info["x"]) == 1
                and len(landmark_info["y"]) == 1
                and len(landmark_info["id"]) == 1
            ), "Invalid format!"
        lm_x_values = np.array(lm_x_values)
        lm_y_values = np.array(lm_y_values)
        x_extent = lm_x_values.max() - lm_x_values.min()
        y_extent = lm_y_values.max() - lm_y_values.min()
        x_center = (lm_x_values.max() + lm_x_values.min()) / 2.
        y_center = (lm_y_values.max() + lm_y_values.min()) / 2.
        x1_final = x_center - x_extent * 0.65
        x2_final = x_center + x_extent * 0.65
        y1_final = y_center - y_extent * 0.65
        y2_final = y_center + y_extent * 0.65
        person.rect = {"x1": x1_final, "x2": x2_final, "y1": y1_final, "y2": y2_final}
        return person

    @classmethod
    def from_new(cls, person_info, conversion_table):
        """Parse a dictionary representation from the PoseTrack18 format."""
        global SCORE_WARNING_EMITTED  # pylint: disable=global-statement
        person = Person(person_info["track_id"])
        try:
            rect_head = {}
            rect_head["x1"] = person_info["bbox_head"][0]
            rect_head["x2"] = person_info["bbox_head"][0] + person_info["bbox_head"][2]
            rect_head["y1"] = person_info["bbox_head"][1]
            rect_head["y2"] = person_info["bbox_head"][1] + person_info["bbox_head"][3]
            person.rect_head = rect_head
        except KeyError:
            person.rect_head = None
        try:
            rect = {}
            rect["x1"] = person_info["bbox"][0]
            rect["x2"] = person_info["bbox"][0] + person_info["bbox"][2]
            rect["y1"] = person_info["bbox"][1]
            rect["y2"] = person_info["bbox"][1] + person_info["bbox"][3]
            person.rect = rect
        except KeyError:
            person.rect = None
        if "score" in person_info.keys():
            person.score = person_info["score"]
        try:
            landmark_scores = person_info["scores"]
        except KeyError:
            landmark_scores = None
            if not SCORE_WARNING_EMITTED:
                LOGGER.warning("No landmark scoring information found!")
                LOGGER.warning("This will not be a valid submission file!")
                SCORE_WARNING_EMITTED = True
        person.landmarks = []
        for landmark_idx, landmark_info in enumerate(
            np.array(person_info["keypoints"]).reshape(len(conversion_table), 3)
        ):
            landmark_idx_can = conversion_table[landmark_idx]
            if landmark_idx_can is not None:
                lm_info = {
                    "y": landmark_info[1],
                    "x": landmark_info[0],
                    "id": landmark_idx_can,
                    "is_visible": landmark_info[2],
                }
                if landmark_scores:
                    lm_info["score"] = landmark_scores[landmark_idx]
                person.landmarks.append(lm_info)
        return person


class Image:

    """An image with annotated people on it."""

    def __init__(self, filename, frame_id):
        self.posetrack_filename = filename
        self.frame_id = frame_id
        self.people = []
        self.ignore_regions = None  # None or tuple of (regions_x, regions_y), each a
        # list of lists of polygon coordinates.

    def to_new(self):
        """
        Return a dictionary representation for the PoseTrack18 format.

        The field 'vid_id' must still be added.
        """
        ret = {
            "file_name": self.posetrack_filename,
            "has_no_densepose": True,
            "is_labeled": (len(self.people) > 0),
            "frame_id": self.frame_id,
            # vid_id and nframes are inserted later.
        }
        if self.ignore_regions:
            ret["ignore_regions_x"] = self.ignore_regions[0]
            ret["ignore_regions_y"] = self.ignore_regions[1]
        return ret

    def to_old(self):
        """
        Return a dictionary representation for the PoseTrack17 format.

        People are added later.
        """
        ret = {"name": self.posetrack_filename}
        if self.ignore_regions:
            ir_list = []
            for plist_x, plist_y in zip(self.ignore_regions[0], self.ignore_regions[1]):
                r_list = []
                for x_val, y_val in zip(plist_x, plist_y):
                    r_list.append({"x": [x_val], "y": [y_val]})
                ir_list.append({"point": r_list})
        else:
            ir_list = None
        imgnum = int(path.basename(self.posetrack_filename).split(".")[0]) + 1
        return ret, ir_list, imgnum

    @classmethod
    def from_old(cls, json_data):
        """Parse a dictionary representation from the PoseTrack17 format."""
        posetrack_filename = json_data["image"][0]["name"]
        assert len(json_data["image"]) == 1, "Invalid format!"
        old_seq_fp = path.basename(path.dirname(posetrack_filename))
        fp_wo_ending = path.basename(posetrack_filename).split(".")[0]
        if "_" in fp_wo_ending:
            fp_wo_ending = fp_wo_ending.split("_")[0]
        old_frame_id = int(fp_wo_ending)
        try:
            frame_id = posetrack18_fname2id(old_seq_fp, old_frame_id)
        except:  # pylint: disable=bare-except
            print("I stumbled over a strange sequence. Maybe you can have a look?")
            import pdb

            pdb.set_trace()  # pylint: disable=no-member
        image = Image(posetrack_filename, frame_id)
        for person_info in json_data["annorect"]:
            image.people.append(Person.from_old(person_info))
        if "ignore_regions" in json_data.keys():
            ignore_regions_x = []
            ignore_regions_y = []
            for ignore_region in json_data["ignore_regions"]:
                x_values = []
                y_values = []
                for point in ignore_region["point"]:
                    x_values.append(point["x"][0])
                    y_values.append(point["y"][0])
                ignore_regions_x.append(x_values)
                ignore_regions_y.append(y_values)
            image.ignore_regions = (ignore_regions_x, ignore_regions_y)
        return image

    @classmethod
    def from_new(cls, track_data, image_id):
        """Parse a dictionary representation from the PoseTrack18 format."""
        image_info = [
            image_info
            for image_info in track_data["images"]
            if image_info["id"] == image_id
        ][0]
        posetrack_filename = image_info["file_name"]
        # license, coco_url, height, width, date_capture, flickr_url, id are lost.
        old_seq_fp = path.basename(path.dirname(posetrack_filename))
        old_frame_id = int(path.basename(posetrack_filename).split(".")[0])
        frame_id = posetrack18_fname2id(old_seq_fp, old_frame_id)
        image = Image(posetrack_filename, frame_id)
        if (
            "ignore_regions_x" in image_info.keys()
            and "ignore_regions_y" in image_info.keys()
        ):
            image.ignore_regions = (
                image_info["ignore_regions_x"],
                image_info["ignore_regions_y"],
            )
        return image


@click.command()
@click.argument(
    "in_fp", type=click.Path(exists=True, readable=True, dir_okay=True, file_okay=True)
)
@click.option(
    "--out_fp",
    type=click.Path(exists=False, writable=True, file_okay=False),
    default="converted",
    help="Write the results to this folder (may not exist). Default: converted.",
)
def cli(in_fp, out_fp="converted"):
    """Convert between PoseTrack18 and PoseTrack17 format."""
    #LOGGER.info("Converting `%s` to `%s`...", in_fp, out_fp)
    if in_fp.endswith(".zip") and path.isfile(in_fp):
        #LOGGER.info("Unzipping...")
        import zipfile
        import tempfile

        unzip_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(in_fp, "r") as zip_ref:
            zip_ref.extractall(unzip_dir)
        in_fp = unzip_dir
        #LOGGER.info("Done.")
    else:
        unzip_dir = None
    if path.isfile(in_fp):
        track_fps = [in_fp]
    else:
        track_fps = sorted(
            [
                path.join(in_fp, track_fp)
                for track_fp in os.listdir(in_fp)
                if track_fp.endswith(".json")
            ]
        )
    #LOGGER.info("Identified %d track files.", len(track_fps))
    assert path.isfile(track_fps[0]), "`%s` is not a file!" % (track_fps[0])
    with open(track_fps[0], "r") as inf:
        first_track = json.load(inf)
    # Determine format.
    old_to_new = False
    if "annolist" in first_track.keys():
        old_to_new = True
        #LOGGER.info("Detected PoseTrack17 format. Converting to 2018...")
    else:
        assert "images" in first_track.keys(), "Unknown image format. :("
        #LOGGER.info("Detected PoseTrack18 format. Converting to 2017...")

    videos = []
    #LOGGER.info("Parsing data...")
    for track_fp in tqdm.tqdm(track_fps):
        with open(track_fp, "r") as inf:
            track_data = json.load(inf)
        if old_to_new:
            videos.extend(Video.from_old(track_data))
        else:
            videos.extend(Video.from_new(track_data))
    #LOGGER.info("Writing data...")
    if not path.exists(out_fp):
        os.mkdir(out_fp)
    for video in tqdm.tqdm(videos):
        target_fp = path.join(
            out_fp, posetrack18_id2fname(video.frames[0].frame_id)[0] + ".json"
        )
        if old_to_new:
            converted_json = video.to_new()
        else:
            converted_json = video.to_old()
        with open(target_fp, "w") as outf:
            json.dump(converted_json, outf, default=json_default)
    if unzip_dir:
        #LOGGER.debug("Deleting temporary directory...")
        os.unlink(unzip_dir)
    #LOGGER.info("Done.")

def convert_videos(track_data):
    """Convert between PoseTrack18 and PoseTrack17 format."""
    if "annolist" in track_data.keys():
        old_to_new = True
        #LOGGER.info("Detected PoseTrack17 format. Converting to 2018...")
    else:
        old_to_new = False
        assert "images" in track_data.keys(), "Unknown image format. :("
        #LOGGER.info("Detected PoseTrack18 format. Converting to 2017...")

    if (old_to_new):
        videos = Video.from_old(track_data)
        videos_converted = [v.to_new() for v in videos]
    else:
        videos = Video.from_new(track_data)
        videos_converted = [v.to_old() for v in videos]
    return videos_converted

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    cli()  # pylint: disable=no-value-for-parameter
