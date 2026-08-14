from datetime import datetime
import io
import os
from pathlib import Path
import subprocess
import sys

from lftomo.util.log import LOG, Logger, log


def test_LOG():
	assert LOG.SILENT == 0
	assert LOG.TIME | LOG.WARN == 12
	assert LOG.STATUS | LOG.DEBUG == 34
	assert LOG.ERROR.color == "red"
	assert str(LOG.INFO) == "INFO"


def attach_test(step, pid):
	assert pid == os.getpid()


def test_log(capfd, tmp_path):
	assert isinstance(log, Logger)
	log_dir = tmp_path / "testlog"
	log.set_logdir(log_dir)
	assert log_dir.exists()

	# Header Test
	log.header()
	out, err = capfd.readouterr()
	assert out == f'{"TYPE":6}|{"STEP":20}|   TIMESTAMP   |MEM USAGE|MEM FREE | STATEMENT \n'
	assert err == ""

	# stdout write test
	log.write(step="LOG TEST", statement="LOG TEST", log_level=LOG.STATUS)
	out, err = capfd.readouterr()
	assert out[:28] == F"STATUS|{'LOG TEST':20}|"
	assert out[-10:] == "LOG TEST\"\n"
	assert err == ""

	# Attach func for testing attach calls.
	log.attach_func(attach_test)

	# stderr write test
	log.write(step="LOG TEST", statement="ERROR TEST", log_level=LOG.ERROR)
	out, err = capfd.readouterr()
	assert err[:28] == F"ERROR |{'LOG TEST':20}|"
	assert err[-12:] == "ERROR TEST\"\n"
	assert out == ""

	# dump tests
	log.dump("Ignore Me", use_color=True)
	out, err = capfd.readouterr()
	assert out == "Ignore Me\n"
	assert err == ""

	log.dump("Still Ignore Me", log_level=LOG.ERROR)
	out, err = capfd.readouterr()
	assert err == "Still Ignore Me\n"
	assert out == ""

	log.footer()
	out, err = capfd.readouterr()
	assert err == ""
	assert out[:28] == F"TIME  |{'COMPLETED':20}|"

	log.footer(error=FileNotFoundError("Ignore Me"))
	out, err = capfd.readouterr()
	assert err == ""
	assert out[:28] == F"TIME  |{'ERRORED':20}|"
	assert out[-12:] == "\"Ignore Me\"\n"
	log.close()


def test_import_construct_and_default_write_create_no_cwd_file(tmp_path, monkeypatch, capfd):
	monkeypatch.chdir(tmp_path)
	local = Logger()
	local.write("Default", "screen only", LOG.STATUS)
	local.close()
	capfd.readouterr()
	assert list(tmp_path.iterdir()) == []

	repository = Path(__file__).resolve().parents[2]
	environment = dict(os.environ)
	environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
		str(repository), environment.get("PYTHONPATH"),
	)))
	subprocess.run(
		[sys.executable, "-c", (
			"from lftomo.util.log import Logger; "
			"logger = Logger(); logger.write('Import', 'screen only'); logger.close()"
		)],
		cwd=tmp_path,
		env=environment,
		check=True,
		capture_output=True,
		text=True,
	)
	assert list(tmp_path.iterdir()) == []


def test_explicit_path_sink_flushes_and_closes(tmp_path):
	path = tmp_path / "operator.log"
	with Logger(
		log_screen={},
		log_files={"operator": (path, LOG.TIME)},
	) as local:
		assert not path.exists()
		local.write("Explicit", "written", LOG.TIME)
		assert path.read_text(encoding="utf-8").endswith('|"written"\n')
	assert path.is_file()


def test_caller_stream_sink_is_flushed_but_not_closed():
	stream = io.StringIO()
	local = Logger(
		log_screen={},
		log_files={"borrowed": (stream, LOG.TIME)},
	)
	local.write("Borrowed", "still owned by caller", LOG.TIME)
	local.close()
	assert not stream.closed
	assert stream.getvalue().endswith('|"still owned by caller"\n')


def test_confirm(monkeypatch, tmp_path):
	new_log = Logger(log_screen={
				"stdout": LOG.STATUS | LOG.TIME | LOG.INFO | LOG.DEBUG,
				"stderr": LOG.ERROR | LOG.WARN
			}, log_files={
				"general": (tmp_path / f"imaging_log_{datetime.now()}.txt", ~(LOG.INFO))
			})

	# Attach func for testing attach calls.
	new_log.attach_func(attach_test)

	# Created Header Test
	new_log.header()

	# Confirmation prompt testing.
	monkeypatch.setattr('sys.stdin', io.StringIO("Y\n"))
	assert new_log.confirm("LOG TEST", "Accept Test")
	monkeypatch.setattr('sys.stdin', io.StringIO("N\n"))
	assert not new_log.confirm("LOG TEST", "Reject Test")

	# General prompt testing
	monkeypatch.setattr('sys.stdin', io.StringIO("My Hat\n"))
	assert new_log.prompt("PROMPT TEST", "What is on my head?") == "My Hat"
	new_log.close()


def test_out_set(capfd, tmp_path):
	new_log = Logger(log_screen={
				"stdout": None,
				"stderr": LOG.ERROR | LOG.WARN
			}, log_files={
				"general": (tmp_path / f"imaging_log_{datetime.now()}.txt", ~(LOG.INFO))
			})

	x = io.StringIO()
	new_log.header(out=x)
	out, err = capfd.readouterr()
	assert x.getvalue() == f'{"TYPE":6}|{"STEP":20}|   TIMESTAMP   |MEM USAGE|MEM FREE | STATEMENT \n'
	assert out == ""
	assert err == ""

	new_log.header()
	out, err = capfd.readouterr()
	assert out == ""
	assert err == f'{"TYPE":6}|{"STEP":20}|   TIMESTAMP   |MEM USAGE|MEM FREE | STATEMENT \n'

	x = io.StringIO()
	new_log.dump("Ignore Me", out=x)
	assert x.getvalue() == "Ignore Me\n"

	x = io.StringIO()
	new_log.write("LOG EXC TEST", "No Error", log_level=LOG.INFO, write_tb=True, out=x)

	print(x.getvalue())

	assert x.getvalue()[:28] == F"INFO  |{'LOG EXC TEST':20}|"
	assert x.getvalue()[-19:] == "\"No Error-tb-None\"\n"
	new_log.close()
