# Runtime image: the released CUDA wheel, pip-free and RunPod-ready.
# Published as ghcr.io/daniel-m-campos/bonsai:{vX.Y.Z-cuda,cuda} by the
# wheels workflow AFTER the GPU pod gate passes; the gate itself boots the
# :candidate-<tag> push of this image, so image start and wheel behavior are
# proven in the same rented session.
#
# Base pinned 12.4 for the same reason as ci.Dockerfile: cu12.8+ base images
# silently fail to start on 12.4-driver RunPod hosts. The wheel carries its
# own static cudart (built against 12.8), so the base's CUDA version is
# irrelevant to it; only the host driver matters (R525+). -base, not -devel:
# nothing is compiled here.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# jammy's python3 is 3.10; the build arg names which wheel from the release
# fan gets baked in.
RUN apt-get update -q && apt-get install -y -q --no-install-recommends \
        python3 python3-pip openssh-server \
    && rm -rf /var/lib/apt/lists/*

COPY dist/ /tmp/dist/
RUN python3 -m pip install --no-cache-dir /tmp/dist/bonsai_gbt-*cp310*.whl \
    && rm -rf /tmp/dist

# Same entrypoint as the ci image: PUBLIC_KEY -> authorized_keys, foreground
# sshd. Makes the image directly usable as a RunPod pod; for local docker
# use, override the command (docker run ... python3).
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
