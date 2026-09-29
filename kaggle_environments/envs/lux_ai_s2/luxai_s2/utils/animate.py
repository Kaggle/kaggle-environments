def animate(imgs, _return=True, fps=10, filename="__temp__.mp4"):
    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(imgs, fps=fps)
    clip.write_videofile(filename, verbose=False, logger=None)
    if _return:
        from IPython.display import Video

        return Video(filename)
