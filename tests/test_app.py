from contour_detect.app import MonocularContourApp


def test_app_pipeline_runs():
    app = MonocularContourApp()
    app.load_frames(["frame-0"])
    result = app.process_current()

    assert result.ok is True
    assert result.contour_length > 0
