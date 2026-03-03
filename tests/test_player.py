from contour_detect.player import ListFrameSource, LoopMode, PlayerController


def test_playback_tick_and_loop():
    player = PlayerController(ListFrameSource(["a", "b"]))
    player.set_loop_mode(LoopMode.ALL)
    player.play()

    assert player.tick() == "b"
    assert player.tick() == "a"


def test_seek_and_step():
    player = PlayerController(ListFrameSource([0, 1, 2]))
    assert player.seek(2) == 2
    assert player.step(-1) == 1
