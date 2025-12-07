from pcoa_app.utils.enums import *
from core.pcoa_app import PCOAApp
from core.pcoa_ai_model import ColorAnalysisModel

if __name__ == "__main__":
    pa = PCOAApp()
    ui = pa.build_ui()
    model = ColorAnalysisModel()
    ui.launch()