"""
tests/test_e2e_app.py

End-to-end AppTest verifying app.py initializes, tabs render, and monitoring metrics compute with zero errors.
"""
import pytest
from streamlit.testing.v1 import AppTest

def test_app_initialization_and_tabs():
    """
    Test that app.py loads, initializes session state, and renders all UI components without exceptions.
    """
    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    
    assert not at.exception, f"App raised an exception: {at.exception}"
    assert at.title[0].value == "📜 Software & AI Patent Infringement Auditor"
    assert len(at.tabs) == 4
    assert len(at.metric) >= 4
