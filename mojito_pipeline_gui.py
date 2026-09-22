"""
Backward-compatible entry point for the Mojito barkeeper GUI.

Prefer ``mojito-barkeeper-gui`` or ``python -m mojito_barkeeper.gui``.
"""

from mojito_barkeeper.gui import MojitoPipelineGUI, main, parse_args

__all__ = ["MojitoPipelineGUI", "main", "parse_args"]

if __name__ == "__main__":
    main()
