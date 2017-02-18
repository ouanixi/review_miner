FROM python:3.5-onbuild

RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

RUN py.test tests/ -v
RUN flake8 app tests

ENTRYPOINT uwsgi --ini ./config/uwsgi/uwsgi.ini