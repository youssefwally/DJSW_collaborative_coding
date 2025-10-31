# Generating the docs

Use [sphinx](http://www.sphinx-doc.org/) structure to update the documentation. 

Add dependencies:

    pip install -r docs/requirements-docs.txt 

Build locally with:

    sphinx-build docs docs/_build

Serve locally with:

    sphinx-autobuild --port 8000 docs docs/_build