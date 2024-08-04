import sys
import os
sys.path.append(os.getcwd())

import torch
import threading

from fedlab.utils import MessageCode
from fedlab.core.server.handler import ServerHandler
from fedlab.core.client import PassiveClientManager, SERIAL_TRAINER, ORDINARY_TRAINER
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.core.network import DistNetwork
from fedlab.utils import Logger

class FedAPPassiveClientManager(PassiveClientManager):
    def __init__(self,
                network: DistNetwork,
                trainer: ModelMaintainer,
                logger: Logger=None
                ):
        super().__init__(network, trainer, logger)

    def main_loop(self):
        """Actions to perform when receiving a new message, including local training.

        Main procedure of each client:
            1. client waits for data from server (PASSIVELY).
            2. after receiving data, client start local model training procedure.
            3. client synchronizes with server actively.
        """
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)

            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:
                id_list, payload, cur_round, rank = payload[0].to(torch.int32), payload[1:-2], payload[-2].to(torch.int32), payload[-1].to(torch.int32)

                # check the trainer type
                if self._trainer.type == SERIAL_TRAINER:
                    self._trainer.local_process(payload=payload, id_list=id_list, cur_round=cur_round, rank=rank)

                elif self._trainer.type == ORDINARY_TRAINER:
                    assert len(id_list) == 1
                    self._trainer.local_process(payload=payload, id=id_list[0])

                self.synchronize()

            else:
                raise ValueError(
                    "Invalid MessageCode {}. Please check MessageCode list.".
                    format(message_code))
            
    def shutdown(self):
        if self._trainer.wandb_logger is not None:
            self._trainer.wandb_logger.run.finish()
        super().shutdown()

class FedAPSynchronousServerManager(SynchronousServerManager):
    def __init__(self,
                network: DistNetwork,
                handler: ServerHandler,
                mode: str = "LOCAL",
                logger: Logger = None):
        super().__init__(network, handler, mode, logger)

    def activate_clients(self):
        """Activate subset of clients to join in one FL round

        Manager will start a new thread to send activation package to chosen clients' process rank.
        The id of clients are obtained from :meth:`handler.sample_clients`. And their communication ranks are are obtained via coordinator.
        """
        self._LOGGER.info("Client activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self._LOGGER.info("Client id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(
                content=[id_list] + downlink_package + [torch.tensor(self._handler.round).to(downlink_package[0].dtype), torch.tensor(rank).to(downlink_package[0].dtype)],
                message_code=MessageCode.ParameterUpdate,
                dst=rank
            )
    
    def main_loop(self):
        """Actions to perform in server when receiving a package from one client.

        Server transmits received package to backend computation handler for aggregation or others
        manipulations.

        Loop:
            1. activate clients for current training round.
            2. listen for message from clients -> transmit received parameters to server handler.

        Note:
            Communication agreements related: user can overwrite this function to customize
            communication agreements. This method is key component connecting behaviors of
            :class:`ServerHandler` and :class:`NetworkManager`.

        Raises:
            Exception: Unexpected :class:`MessageCode`.
        """
        while self._handler.if_stop is not True:
            activator = threading.Thread(target=self.activate_clients)
            activator.start()

            while True:
                sender_rank, message_code, payload = self._network.recv()
                if message_code == MessageCode.ParameterUpdate:
                    if self._handler.load(payload):
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))
                
    def shutdown(self):
        if self._handler.wandb_logger is not None:
            self._handler.wandb_logger.run.finish()
        super().shutdown()