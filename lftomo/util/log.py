"""Canonical side-effect-free operator logging for lftomo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntFlag
from pathlib import Path
from sys import exc_info, stdout
import threading
from typing import TextIO

import click
import psutil


class LOG(IntFlag):
	SILENT = 0
	ERROR = 1
	STATUS = 2
	TIME = 4
	WARN = 8
	INFO = 16
	DEBUG = 32

	def __init__(self, log_level):
		self.log_level = log_level
		self.__colors = {
			0: "black", 1: "red", 2: "green", 4: "cyan",
			8: "yellow", 16: "white", 32: "magenta",
		}

	def __str__(self):
		return str(self.name)

	@property
	def color(self):
		return self.__colors[self.value]


_DEFAULT_FILE_LEVELS = LOG.ERROR | LOG.STATUS | LOG.TIME | LOG.WARN


@dataclass
class _LogSink:
	"""One explicitly configured path-owned or caller-owned output sink."""

	target: Path | TextIO
	levels: LOG
	stream: TextIO | None = None
	owned: bool = False

	def _open(self) -> TextIO:
		if self.stream is not None:
			return self.stream
		if hasattr(self.target, "write"):
			self.stream = self.target
			self.owned = False
		else:
			path = Path(self.target)
			path.parent.mkdir(parents=True, exist_ok=True)
			self.stream = path.open("a", encoding="utf-8")
			self.owned = True
		return self.stream

	def write(self, message: str) -> None:
		stream = self._open()
		click.echo(message, file=stream)
		stream.flush()

	def flush(self) -> None:
		if self.stream is not None:
			self.stream.flush()

	def close(self) -> None:
		if self.stream is None:
			return
		self.stream.flush()
		if self.owned:
			self.stream.close()
			self.stream = None
			self.owned = False


class Logger:
	"""Format human operator logs and route them to explicitly selected sinks.

	The default logger writes selected severities to stdout/stderr and owns no
	file. A path supplied through ``log_files`` or :meth:`set_logdir` is opened
	lazily and owned by the logger. A caller-provided stream is borrowed: it is
	flushed by :meth:`flush` but never closed by :meth:`close`.

	Memory columns are observations sampled when the log line is formatted. They
	do not attribute memory to a named Stage or execution segment.
	"""

	def __init__(self, log_screen=None, log_files=None):
		self.script_start = datetime.now()
		self.__attached_funcs = []
		self.__log_screen = (
			{
				"stdout": LOG.STATUS | LOG.TIME | LOG.WARN | LOG.INFO,
				"stderr": LOG.ERROR,
			}
			if log_screen is None else dict(log_screen)
		)
		self.__logs: dict[str, _LogSink] = {}
		for name, (target, levels) in ({} if log_files is None else log_files).items():
			self.__logs[name] = _LogSink(target, levels)
		self.__pid = psutil.Process().pid
		self.__lock = threading.RLock()

	def set_logdir(self, logdir):
		"""Explicitly opt into one owned general log under ``logdir``."""

		root = Path(logdir)
		root.mkdir(parents=True, exist_ok=True)
		stamp = self.script_start.strftime("%Y%m%d_%H%M%S.%f")
		with self.__lock:
			prior = self.__logs.get("general")
			levels = _DEFAULT_FILE_LEVELS if prior is None else prior.levels
			if prior is not None:
				prior.close()
			self.__logs["general"] = _LogSink(
				root / f"imaging_log_{stamp}.txt", levels,
			)

	def flush(self) -> None:
		"""Flush configured file and borrowed-stream sinks."""

		with self.__lock:
			for sink in self.__logs.values():
				sink.flush()

	def close(self) -> None:
		"""Close logger-owned files without closing caller-provided streams."""

		with self.__lock:
			for sink in self.__logs.values():
				sink.close()

	def __enter__(self):
		return self

	def __exit__(self, *_args):
		self.close()
		return False

	def header(self, out=None):
		"""Write the explanatory header to the selected operator sinks."""

		header_message = f'{"TYPE":6}|{"STEP":20}|   TIMESTAMP   |MEM USAGE|MEM FREE | STATEMENT '
		self.__special_write(header_message, out)

	def footer(self, out=None, error=None):
		step = "COMPLETED" if error is None else "ERRORED"
		statement = "" if error is None else f"{error}"
		self.__special_write(self.__log_message(step, statement), out)

	def __log_message(self, step: str, statement: str = "", log_level: LOG = LOG.TIME):
		styled_type = click.style(f"{log_level.name:6}", log_level.color)
		return (
			f"{styled_type}"
			f"|{step[:20]:20}"
			f"|{str(datetime.now() - self.script_start).zfill(15)}"
			f"|{psutil.Process().memory_info().vms // 1024 ** 3:07.2f}GB"
			f"|{psutil.virtual_memory().available // 1024 ** 3:07.2f}GB"
			f'|"{statement}"'
		)

	def __special_write(self, message, out=None):
		with self.__lock:
			if out is not None:
				click.echo(message, file=out)
				out.flush()
				return
			if self.__log_screen.get("stdout") is not None:
				click.echo(message)
			elif self.__log_screen.get("stderr") is not None:
				click.echo(message, err=True)
			for sink in self.__logs.values():
				sink.write(message)

	def __multi_write(self, message, log_level):
		with self.__lock:
			screen_levels = LOG.SILENT
			for levels in self.__log_screen.values():
				if levels is not None:
					screen_levels |= levels
			if log_level & screen_levels:
				stderr_levels = self.__log_screen.get("stderr") or LOG.SILENT
				click.echo(message, err=bool(log_level & stderr_levels))
			for sink in self.__logs.values():
				if log_level & sink.levels:
					sink.write(message)

	def write(
		self,
		step: str,
		statement: str = "",
		log_level: LOG = LOG.TIME,
		out=None,
		write_tb=False,
	):
		"""Write one formatted human log line.

		``out`` is a borrowed one-call stream override. Structured experiment
		``ProgressEvent`` records remain a separate machine-readable channel.
		"""

		for func in self.__attached_funcs:
			func(step, self.__pid)
		if write_tb:
			_, _, last_traceback = exc_info()
			statement += f"-tb-{last_traceback}"
		log_line = self.__log_message(step, statement, log_level)
		if out is not None:
			click.echo(log_line, file=out)
			out.flush()
		else:
			self.__multi_write(log_line, log_level)

	def confirm(
		self,
		step: str,
		statement: str = "",
		log_level: LOG = LOG.TIME,
		out: TextIO = stdout,
	):
		"""Display a confirmation prompt in the canonical log format."""

		for func in self.__attached_funcs:
			func(step, self.__pid)
		return click.confirm(
			self.__log_message(step, statement, log_level),
			err=(log_level == LOG.ERROR),
		)

	def prompt(
		self,
		step: str,
		statement: str = "",
		log_level: LOG = LOG.TIME,
		out: TextIO = stdout,
		default=None,
	):
		"""Display a value prompt in the canonical log format."""

		for func in self.__attached_funcs:
			func(step, self.__pid)
		return click.prompt(
			self.__log_message(step, statement, log_level),
			err=(log_level == LOG.ERROR),
			default=default,
		)

	def attach_func(self, func: callable):
		"""Attach a callback invoked with the log step and process ID."""

		if func not in self.__attached_funcs:
			self.__attached_funcs.append(func)

	def dump(
		self,
		statement: str,
		log_level: LOG = LOG.INFO,
		out=None,
		use_color=False,
	):
		"""Write a line without type, timing, or resource columns."""

		if use_color:
			statement = click.style(statement, log_level.color)
		if out is not None:
			click.echo(statement, file=out)
			out.flush()
		else:
			self.__multi_write(statement, log_level)


log = Logger()
