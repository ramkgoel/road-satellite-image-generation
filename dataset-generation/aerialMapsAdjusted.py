import pandas as pd
import datasets
import os
from random import randint

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

METADATA_URL = "./prompt.json"

IMAGES_URL = "./target.zip"

CONDITIONING_IMAGES_URL = "./source.zip"

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)


class AerialMapsAdjusted(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_path = dl_manager.download(METADATA_URL)
        images_dir = dl_manager.download_and_extract(IMAGES_URL)
        conditioning_images_dir = dl_manager.download_and_extract(
            CONDITIONING_IMAGES_URL
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "conditioning_images_dir": conditioning_images_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, conditioning_images_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            # randomly remove the prompt somtimes
            k = randint(0, 5)
            text = "" if k%2 == 0 else row["prompt"]

            image_path = row["target"]
            image_path = os.path.join(images_dir, image_path)
            image = open(image_path, "rb").read()

            conditioning_image_path = row["source"]
            conditioning_image_path = os.path.join(
                conditioning_images_dir, row["source"]
            )
            conditioning_image = open(conditioning_image_path, "rb").read()

            yield row["target"], {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }