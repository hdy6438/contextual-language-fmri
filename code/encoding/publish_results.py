from time import sleep


def publish_directory(stage, target):
    """Move a complete output directory after transient filesystem locks clear."""
    for delay in (0.25, 0.5, 1, 2, 4, 8, 12):
        try:
            stage.rename(target)
        except PermissionError:
            sleep(delay)
        else:
            return
    stage.rename(target)
