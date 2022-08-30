# MABe 2022 submission of team IRLAB for the Mouse Triplets Video Data challenge

## Steps to reproduce our results (except for random variation)
1. Download the challenge data to `/data/behavior-representation/` following the instructions at https://sites.google.com/view/mabe22/home.
    * Extract the submission video data to `/data/behavior-representation/videos/full_size/submission/`.

2. Build a docker image using the Dockerfile: `docker build . -t mabe_2022`.
3. Enter the docker container with `docker run -it --gpus '"device=1"' --shm-size 2g -v SRC_DIR/mabe_2022:/app -v /data/behavior-representation/:/data/behavior-representation -e PYTHONPATH=/app -w /app mabe_2022:latest bash`.
4. Embed all frames with [BEiT](https://huggingface.co/microsoft/beit-large-patch16-512): `python3 utils/embed_frames_beit.py`. This takes ~2.5 days on an Nvidia RTX 3080 GPU.
5. Run `utils/average_motion.py`. This computes a measure of the amount of motion in each frame, based on the keypoints.
6. Run `utils/train_simclr_model.py` to train a SimCLR model and use it to compute an embedding for each frame.
7. Run `utils/handcrafted_geometries.py` to compute a bunch of handcrafted features based on the keypoints.
8. Run `combine_embeddings.ipynb` to combine BEiT embeddings, SimCLR embeddings and handcrafted features with a weighted PCA transform.
9. Run `append_mean_beit_pca.ipynb` to exchange the last 8 dimensions of the previous PCA with the first 8 PCA components of BEiT, averaged over each video snippet.