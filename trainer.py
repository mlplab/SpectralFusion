# coding: utf-8


import os
import sys
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
from evaluate import PSNRMetrics, SAMMetrics, None_Evaluate
from pytorch_ssim import SSIM
from utils import normalize


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# if device == 'cuda':
#     torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=8)


class Trainer(object):

    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, *args, scheduler=None,
                 callbacks=None, device: str='cpu', evaluate_flg: bool=False, **kwargs):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.device = device
        self.use_amp = kwargs.get('use_amp', False)
        self.psnr = kwargs.get('psnr', None_Evaluate())  # SNRMetrics().eval())
        self.sam = kwargs.get('sam', None_Evaluate())  # SAMMetrics().eval())
        self.ssim = kwargs.get('ssim', None_Evaluate())  # SSIM().eval()
        self.colab_mode = kwargs.get('colab_mode', False)
        self.evaluate_flg = evaluate_flg
        self.scaler = torch.cuda.amp.GradScaler()

    def train(self, epochs: int, train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: torch.utils.data.DataLoader, init_epoch: int=0) -> (np.ndarray, np.ndarray):

        if self.colab_mode is False:
            _, columns = os.popen('stty size', 'r').read().split()
            columns = int(columns)
        else:
            columns = 200
        train_output = []
        val_output = []

        for epoch in range(init_epoch, epochs):
            dt_now = datetime.now()
            print(dt_now)
            self.model.train()
            mode = 'Train'
            train_loss = []
            val_loss = []
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            train_show_mean = self._all_step(train_dataloader, mode=mode, desc_str=desc_str, columns=columns)
            train_output.append(train_show_mean)

            self.model.eval()
            mode = 'Val'
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            val_show_mean = self._all_step(val_dataloader, mode=mode, desc_str=desc_str, columns=columns)
            val_output.append(val_show_mean)
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.model, epoch, loss=train_show_mean,
                                      val_loss=val_show_mean, save=True,
                                      device=self.device, optim=self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            print('=' * int(columns))

        train_output = np.array(train_output)
        val_output = np.array(val_output)
        return train_output, val_output

    def _trans_data(self, data: torch.Tensor) -> torch.Tensor:
        if isinstance(data, (list, tuple)):
            return [x.to(self.device) for x in data]
        elif isinstance(data, (dict)):
            return {key: value.to(self.device) for key, value in data.items()}
        else:
            return data.to(self.device)

    def _step(self, inputs: torch.Tensor, labels: torch.Tensor,
              train: bool=True) -> (torch.Tensor, torch.Tensor):
        with torch.cuda.amp.autocast(self.use_amp):
            if isinstance(inputs, (list, tuple)):
                output = self.model(*inputs)
            elif isinstance(inputs, (dict)):
                output = self.model(**inputs)
            else:
                output = self.model(inputs)
            if isinstance(labels, dict) and isinstance(output, torch.Tensor):
                labels = labels['hsi']
            loss = self.criterion(output, labels)
        if train is True:
            if self.device == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()
        if isinstance(output, (list, tuple)):
            hsi = output[-1]
        else:
            hsi = output
        return loss, hsi

    def _step_show(self, pbar, *args, **kwargs) -> None:
        if self.device == 'cuda':
            kwargs['Allocate'] = f'{torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB'
            kwargs['Cache'] = f'{torch.cuda.memory_reserved(0) / 1024 ** 3:.3f}GB'
        pbar.set_postfix(kwargs)
        return self

    def _evaluate(self, output: torch.Tensor, label: torch.Tensor) -> (float, float, float):
        output = output.float().to(self.device)
        output = torch.clamp(output, 0., 1.)
        if isinstance(label, (list, tuple)):
            hsi_label = label[-1]
        elif isinstance(label, (dict)):
            hsi_label = label['hsi']
        else:
            hsi_label = label
        label = torch.clamp(hsi_label, 0., 1.)
        return [self.psnr(label, output).item(),
                self.ssim(label, output).item(),
                self.sam(label, output).item()]

    def _all_step(self, dataloader, mode: str, desc_str: str, columns: int) -> np.ndarray:
        step_loss = []
        step_eval = []
        with tqdm(dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
            for i, (inputs, labels) in enumerate(pbar):
                inputs = self._trans_data(inputs)
                labels = self._trans_data(labels)
                if mode.lower() == 'train':
                    loss, output = self._step(inputs, labels)
                elif mode.lower() == 'val':
                    with torch.no_grad():
                        loss, output = self._step(inputs, labels, train=False)
                if torch.any(torch.isnan(loss)):
                    print('isnan')
                    sys.exit(1)
                step_loss.append(loss.item())
                show_loss = np.mean(step_loss)
                step_eval.append(self._evaluate(output, labels))
                show_mean = np.mean(step_eval, axis=0)
                # if self.evaluate_flg:
                evaluate = [f'{show_mean[0]:.7f}', f'{show_mean[1]:.7f}', f'{show_mean[2]:.7f}']
                self._step_show(pbar, Loss=f'{show_loss:.7f}', Evaluate=evaluate)
                torch.cuda.empty_cache()
        show_loss, show_mean = np.mean(step_loss), np.mean(step_eval, axis=0)
        show_mean = np.insert(show_mean, 0, show_loss)
        return show_mean


class GANTrainer(Trainer):

    def __init__(self, Gmodel, Dmodel, Gcriterion, Dcriterion,
                 Goptim, Doptim, *args, batch_size=64, scheduler=None,
                 callbacks=None, device: str='cpu', evaluate_flg: bool=False, **kwargs):

        super().__init__(None, None, None, scheduler=scheduler, callbacks=callbacks, 
                         device=device, evaluate_flg=evaluate_flg, **kwargs)
        self.Gmodel = Gmodel
        self.Gcriterion = Gcriterion
        self.Goptimizer = Goptim
        self.Dmodel = Dmodel
        self.Dcriterion = Dcriterion
        self.Doptimizer = Doptim
        shape = kwargs.get('shape', (batch_size, 1))
        self.zeros = torch.zeros(shape).to(device)
        self.ones = torch.ones(shape).to(device)
        self.fake_img_criterion = torch.nn.MSELoss().to(device)

    def train(self, epochs, train_dataloader, val_dataloader, init_epoch=None):

        if init_epoch is None:
            init_epoch = 0
        elif isinstance(init_epoch, int):
            assert 'Please enter int to init_epochs'

        if self.colab_mode is False:
            _, columns = os.popen('stty size', 'r').read().split()
            columns = int(columns)
        else:
            columns = 200
        train_output = []
        val_output = []
        train_output_loss = []
        val_output_loss = []

        for epoch in range(init_epoch, epochs):
            dt_now = datetime.now()
            print(dt_now)
            self.Gmodel.train()
            self.Dmodel.train()
            mode = 'Train'
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            train_show_mean = self._reconst_step(train_dataloader, mode, desc_str, columns)
            train_output.append(train_show_mean)

            mode = 'Val'
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            self.Gmodel.eval()
            self.Dmodel.eval()
            val_show_mean = self._reconst_step(val_dataloader, mode, desc_str, columns)
            val_output.append(val_show_mean)
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.Gmodel, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True, device=device, optim=self.Goptimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            print('-' * int(columns))

        train_output = np.array(train_output)
        val_output = np.array(val_output)
        return train_output, val_output

    def predict(self, inputs, labels):
        output = self.Gmodel(inputs)
        loss = self.fake_img_criterion(output, labels)
        return loss, output

    def _step_G(self, inputs, labels, train=True):
        if train is True:
            self.Goptimizer.zero_grad()
            self.Doptimizer.zero_grad()
        bs = inputs.shape[0]
        fake_img = self.Gmodel(inputs)
        pred_fake = self.Dmodel(fake_img)
        loss = self.Gcriterion(pred_fake, self.ones[:bs])
        show_loss = self.fake_img_criterion(fake_img, labels)
        Gloss = loss + 200 * show_loss
        if train is True:
            Gloss.backward()
            self.Goptimizer.step()
        return Gloss, fake_img, show_loss

    def _step_D(self, inputs, labels, train=True):
        if train is True:
            self.Goptimizer.zero_grad()
            self.Doptimizer.zero_grad()
        bs = inputs.shape[0]
        pred_real = self.Dmodel(labels)
        real_loss = self.Dcriterion(pred_real, self.ones[:bs])
        fake_img = self.Gmodel(inputs)
        pred_fake = self.Dmodel(fake_img)
        fake_loss = self.Dcriterion(pred_fake, self.zeros[:bs])
        if train is True:
            real_loss.backward()
            fake_loss.backward()
            self.Doptimizer.step()
        loss = real_loss + fake_loss
        return loss

    def _reconst_step(self, dataloader, mode: str, desc_str: str, columns: int) -> np.ndarray:
        step_loss, step_eval = [], []
        if mode.lower() == 'train':
            step_Dloss, step_Deval = [], []
        with tqdm(dataloader, desc=desc_str, ncols=columns, unit='step', ascii=True) as pbar:
            for i, (inputs, labels) in enumerate(pbar):
                inputs = self._trans_data(inputs)
                labels = self._trans_data(labels)
                if mode.lower() == 'train':
                    Gloss, output, loss = self._step_G(inputs, labels)
                    Dloss = self._step_D(inputs, labels)
                elif mode.lower() == 'val':
                    with torch.no_grad():
                        loss, output = self.predict(inputs, labels)
                step_loss.append(loss.item())
                show_loss = np.mean(step_loss)
                step_eval.append(self._evaluate(output, labels))
                show_mean = np.mean(step_eval, axis=0)
                evaluate = [f'{show_mean[0]:.5f}', f'{show_mean[1]:.5f}', f'{show_mean[2]:.5f}']
                if mode.lower() == 'train':
                    step_Dloss.append(Dloss.item())
                    show_Dloss = np.mean(step_Dloss)
                    self._step_show(pbar, Loss=f'{show_loss:.5f}', DLoss=f'{show_Dloss:.5f}', Evaluate=evaluate)
                elif mode.lower() == 'val':
                    self._step_show(pbar, Loss=f'{show_loss:.5f}', Evaluate=evaluate)
                torch.cuda.empty_cache()
        show_mean = np.insert(show_mean, 0, show_loss)
        return show_mean


class RefineTrainer(Trainer):

    def __init__(self, model, criterion, optimizer, reconst_model,
                 scheduler=None, callbacks=None, device: str='cpu', 
                 evaluate_flg: bool=False, **kwargs):
        super().__init__(model, criterion, optimizer, scheduler=scheduler, callbacks=callbacks, 
                         device=device, evaluate_flg=evaluate_flg, **kwargs)
        self.reconst_model = reconst_model.eval()

    def _step(self, inputs, labels, train=True):
        if train is True:
            self.optimizer.zero_grad()
        with torch.no_grad():
            inputs = self.reconst_model(inputs)
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        if train is True:
            loss.backward()
            self.optimizer.step()
        return loss, output
