from core.base_classes import *
from core.enums import *

if __name__ == "__main__":
    pa = PCOAApp()
    ui = pa.build_ui()
    ui.launch()