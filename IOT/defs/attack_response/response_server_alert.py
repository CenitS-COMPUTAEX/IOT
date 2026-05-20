import os
import socket
import tempfile
import time
from shutil import make_archive

import paramiko as paramiko
from paramiko.client import SSHClient
from paramiko.ssh_exception import SSHException

from IOT.data import dataset_operations
from IOT.defs.config.config import Config as Cfg
from IOT.defs.constants import Constants as Cst

from IOT.defs.attack_response.attack_response import AttackResponse
from IOT.defs.model_prediction import ModelPrediction


class ResponseServerAlert(AttackResponse):
	"""
	Attack response that connects to a remote server and sends an alert
	"""

	ssh: SSHClient
	group_amount: int

	def __init__(self, group_amount: int):
		# Create SSH client
		self.ssh = paramiko.SSHClient()
		self.ssh.load_system_host_keys()
		self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

		self.group_amount = group_amount

		# Stop Paramiko from printing exceptions to the console. We'll catch them and deal with them if required.
		# noinspection PyUnresolvedReferences
		paramiko.util.log_to_file(os.devnull)

	def run(self, last_prediction: ModelPrediction, device_id: str, real_response: bool):
		tries = Cfg.get().max_connection_tries
		while True:
			try:
				self._connect()
				break
			except socket.timeout:
				print("Warning: Connection to the alert server timed out")
				return
			except (SSHException, socket.error):
				tries -= 1
				print("Warning: Failed to estabilish SSH connection to the alert server. Attempts left: " + str(tries))
				if tries > 0:
					time.sleep(0.1)
				else:
					return

		# Create a temporary directory to store the files that will be sent to the server
		with tempfile.TemporaryDirectory() as tmp_folder:
			os.mkdir(os.path.join(tmp_folder, Cst.ALERT_FOLDER))
			with open(os.path.join(tmp_folder, Cst.ALERT_FOLDER, Cst.ALERT_INFO_FILE), "w") as f:
				f.write("Device: " + device_id + "\n")
				f.write("Group amount: " + str(self.group_amount) + "\n")
				f.write("Measurement delay: " + str(dataset_operations.measurement_delay) + "\n")
				f.write("Prediction time: " + str(last_prediction.time) + "\n")
				f.write("Real alert: " + str(real_response))

			with open(os.path.join(tmp_folder, Cst.ALERT_FOLDER, Cst.ALERT_PREDICTION_FILE), "w") as f:
				f.write(last_prediction.header + "\n" + last_prediction.data)

			# Compress everything into a zip file to send it over to the server
			make_archive((os.path.join(tmp_folder, Cst.ALERT_FILE)), 'zip', os.path.join(tmp_folder, Cst.ALERT_FOLDER))
			with open(os.path.join(tmp_folder, Cst.ALERT_FILE + ".zip"), "rb") as f:
				file_bytes = f.read()

				tries = Cfg.get().max_connection_tries
				while True:
					try:
						channel = self.ssh.get_transport().open_session(timeout=Cfg.get().alert_server_timeout)
						channel.sendall(file_bytes)
						break
					except socket.timeout:
						print("Warning: Timeout when trying to send alert to the alert server")
						break
					except (SSHException, socket.error):
						tries -= 1
						print("Warning: Failed to send alert to the alert server. Attempts left: " + str(tries))
						if tries > 0:
							time.sleep(0.1)
						else:
							break
				self.ssh.close()

	def _connect(self):
		"""
		Attempts to connect to the alert server
		"""
		if Cfg.get().alert_server_key_file is not None:
			self.ssh.connect(Cfg.get().alert_server_ip, Cfg.get().alert_server_port,
				Cfg.get().alert_server_username, key_filename=Cfg.get().alert_server_key_file,
				timeout=Cfg.get().alert_server_timeout)
		elif Cfg.get().alert_server_password != "":
			self.ssh.connect(Cfg.get().alert_server_ip, Cfg.get().alert_server_port,
				Cfg.get().alert_server_username, Cfg.get().alert_server_password,
				timeout=Cfg.get().alert_server_timeout)
		else:
			# Workaround to connect without authentication
			# https://github.com/paramiko/paramiko/issues/890#issuecomment-906893725
			try:
				self.ssh.connect(Cfg.get().alert_server_ip, Cfg.get().alert_server_port,
				timeout=Cfg.get().alert_server_timeout)
			except SSHException:
				self.ssh.get_transport().auth_none(Cfg.get().alert_server_username)
