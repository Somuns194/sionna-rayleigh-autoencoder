"""
End-to-End Autoencoder Communication over Rayleigh Fading Channel
=================================================================
Based on Sionna's Autoencoder.ipynb, extended with:
  1. Rayleigh block fading channel (y = h*x + n)
  2. Perfect CSI vs. imperfect CSI scenarios
  3. Model mismatch analysis: BCE trained with perfect CSI, tested with imperfect CSI
  4. RL robustness: RL trained with imperfect CSI, tested with imperfect CSI

Windows compatibility: no torch.compile, compile_mode=None for sim_ber
"""

import os
import pickle
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo

import sionna.phy
from sionna.phy import Block
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no, expand_to_rank, sim_ber
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource

sionna.phy.config.seed = 42
device = sionna.phy.config.device
print(f"Device: {device}")

# ============================================================
# Simulation Parameters
# ============================================================
num_bits_per_symbol = 6       # 64-QAM
modulation_order = 2**num_bits_per_symbol
coderate = 0.5
n = 1500
num_symbols_per_codeword = n // num_bits_per_symbol
k = int(n * coderate)

# AWGN parameters (original)
awgn_ebno_db_min = 4.0
awgn_ebno_db_max = 8.0

# Rayleigh parameters (extended SNR range)
rayleigh_ebno_db_min = 10.0
rayleigh_ebno_db_max = 25.0

# Training configuration
num_training_iterations_conventional = 5000
num_training_iterations_rl_alt = 3500
num_training_iterations_rl_finetuning = 1500
training_batch_size = 128
rl_perturbation_var = 0.01

# CSI configuration
csi_error_var = 0.1  # NMSE = -10 dB

# Weight paths
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)
model_weights_path_awgn_conv = os.path.join(weights_dir, "awgn_conv")
model_weights_path_awgn_rl = os.path.join(weights_dir, "awgn_rl")
model_weights_path_rayleigh_conv = os.path.join(weights_dir, "rayleigh_conv")
model_weights_path_rayleigh_rl = os.path.join(weights_dir, "rayleigh_rl")
model_weights_path_rayleigh_rl_imperfect = os.path.join(weights_dir, "rayleigh_rl_imperfect")

results_filename = "rayleigh_results.pkl"


# ============================================================
# Neural Demapper (unchanged from original)
# ============================================================
class NeuralDemapper(nn.Module):
    def __init__(self):
        super().__init__()
        self._dense_1 = nn.Linear(3, 128)
        self._dense_2 = nn.Linear(128, 128)
        self._dense_3 = nn.Linear(128, num_bits_per_symbol)

    def forward(self, y, no):
        no_db = torch.log10(no)
        no_db = no_db.expand(-1, num_symbols_per_codeword)
        z = torch.stack([y.real, y.imag, no_db], dim=2)
        llr = F.relu(self._dense_1(z))
        llr = F.relu(self._dense_2(llr))
        llr = self._dense_3(llr)
        return llr


# ============================================================
# Rayleigh Block Fading Channel Helper
# ============================================================
def apply_rayleigh_fading(x, no, csi_mode='perfect', csi_error_var=0.0):
    """Apply Rayleigh block fading: y = h*x + n.
    
    Args:
        x: [batch_size, num_symbols] complex tensor
        no: [batch_size, 1] noise variance (float)
        csi_mode: 'perfect' or 'imperfect'
        csi_error_var: variance of CSI estimation error
    
    Returns:
        y_eq: equalized received signal [batch_size, num_symbols]
        no_eff: effective noise variance [batch_size, 1]
    """
    batch_size = x.shape[0]
    
    # Generate block fading coefficient h ~ CN(0, 1)
    h = torch.complex(
        torch.randn(batch_size, 1, device=x.device, dtype=x.real.dtype),
        torch.randn(batch_size, 1, device=x.device, dtype=x.real.dtype)
    ) / (2**0.5)
    
    # Apply fading: y = h*x + n
    # Use Sionna's complex_normal for noise generation
    awgn = AWGN()
    x_faded = h * x
    y = awgn(x_faded, no)
    
    # Generate CSI
    if csi_mode == 'perfect':
        h_hat = h
    elif csi_mode == 'imperfect':
        e = torch.complex(
            torch.randn(batch_size, 1, device=x.device, dtype=x.real.dtype),
            torch.randn(batch_size, 1, device=x.device, dtype=x.real.dtype)
        ) / (2**0.5) * (csi_error_var ** 0.5)
        h_hat = h + e
    else:
        raise ValueError(f"Unknown csi_mode: {csi_mode}")
    
    # Equalization: y_eq = y / h_hat
    y_eq = y / h_hat
    
    # Effective noise variance: no_eff = no / |h_hat|^2
    no_eff = no / (h_hat.abs() ** 2)
    
    return y_eq, no_eff


# ============================================================
# E2E System: Conventional Training (BCE)
# ============================================================
class E2ESystemConventionalTraining(nn.Module):
    def __init__(self, training, channel_type='awgn', csi_mode='perfect'):
        super().__init__()
        self._training = training
        self._channel_type = channel_type
        self._csi_mode = csi_mode
        
        # Transmitter
        self._binary_source = BinarySource()
        if not self._training:
            self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        
        qam_points = Constellation("qam", num_bits_per_symbol).points
        self.points_r = nn.Parameter(qam_points.real.clone())
        self.points_i = nn.Parameter(qam_points.imag.clone())
        self.constellation = Constellation("custom", num_bits_per_symbol,
                                           points=torch.complex(self.points_r, self.points_i),
                                           normalize=True, center=True)
        self._mapper = Mapper(constellation=self.constellation)
        
        # Channel
        self._channel = AWGN()
        
        # Receiver
        self._demapper = NeuralDemapper()
        if not self._training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    def forward(self, batch_size, ebno_db):
        self.constellation.points = torch.complex(self.points_r, self.points_i)
        
        if ebno_db.dim() == 0:
            ebno_db = ebno_db.expand(batch_size)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)
        
        # Transmitter
        if self._training:
            c = self._binary_source([batch_size, n])
        else:
            b = self._binary_source([batch_size, k])
            c = self._encoder(b)
        x = self._mapper(c)
        
        # Channel
        if self._channel_type == 'awgn':
            y = self._channel(x, no)
            no_eff = no
        elif self._channel_type == 'rayleigh':
            y, no_eff = apply_rayleigh_fading(x, no, self._csi_mode, csi_error_var)
        else:
            raise ValueError(f"Unknown channel_type: {self._channel_type}")
        
        # Receiver
        llr = self._demapper(y, no_eff)
        llr = llr.reshape(batch_size, n)
        
        if self._training:
            loss = F.binary_cross_entropy_with_logits(llr, c)
            return loss
        else:
            b_hat = self._decoder(llr)
            return b, b_hat


# ============================================================
# E2E System: RL-based Training
# ============================================================
class E2ESystemRLTraining(nn.Module):
    def __init__(self, training, channel_type='awgn', csi_mode='perfect'):
        super().__init__()
        self._training = training
        self._channel_type = channel_type
        self._csi_mode = csi_mode
        
        # Transmitter
        self._binary_source = BinarySource()
        if not self._training:
            self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        
        qam_points = Constellation("qam", num_bits_per_symbol).points
        self.points_r = nn.Parameter(qam_points.real.clone())
        self.points_i = nn.Parameter(qam_points.imag.clone())
        self.constellation = Constellation("custom", num_bits_per_symbol,
                                           points=torch.complex(self.points_r, self.points_i),
                                           normalize=True, center=True)
        self._mapper = Mapper(constellation=self.constellation)
        
        # Channel
        self._channel = AWGN()
        
        # Receiver
        self._demapper = NeuralDemapper()
        if not self._training:
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    def forward(self, batch_size, ebno_db, perturbation_variance=0.0):
        if ebno_db.dim() == 0:
            ebno_db = ebno_db.expand(batch_size)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)
        
        # Transmitter
        if self._training:
            c = self._binary_source([batch_size, n])
        else:
            b = self._binary_source([batch_size, k])
            c = self._encoder(b)
        
        self.constellation.points = torch.complex(self.points_r, self.points_i)
        x = self._mapper(c)
        
        # Perturbation (for RL exploration)
        std = (0.5 * perturbation_variance) ** 0.5
        epsilon_r = torch.randn(x.shape, device=x.device, dtype=x.real.dtype) * std
        epsilon_i = torch.randn(x.shape, device=x.device, dtype=x.real.dtype) * std
        epsilon = torch.complex(epsilon_r, epsilon_i)
        x_p = x + epsilon
        
        # Channel
        if self._channel_type == 'awgn':
            y = self._channel(x_p, no)
            no_eff = no
        elif self._channel_type == 'rayleigh':
            y, no_eff = apply_rayleigh_fading(x_p, no, self._csi_mode, csi_error_var)
        else:
            raise ValueError(f"Unknown channel_type: {self._channel_type}")
        
        y = y.detach()  # Stop gradient (non-differentiable channel)
        
        # Receiver
        llr = self._demapper(y, no_eff)
        
        if self._training:
            c_reshaped = c.reshape(-1, num_symbols_per_codeword, num_bits_per_symbol)
            bce = F.binary_cross_entropy_with_logits(llr, c_reshaped, reduction='none').mean(dim=2)
            rx_loss = bce.mean()
            bce = bce.detach()
            x_p = x_p.detach()
            p = x_p - x
            tx_loss = p.real.square() + p.imag.square()
            tx_loss = -bce * tx_loss / rl_perturbation_var
            tx_loss = tx_loss.mean()
            return tx_loss, rx_loss
        else:
            llr = llr.reshape(-1, n)
            b_hat = self._decoder(llr)
            return b, b_hat


# ============================================================
# Baseline (standard QAM + conventional demapper)
# ============================================================
class Baseline(nn.Module):
    def __init__(self, channel_type='awgn', csi_mode='perfect'):
        super().__init__()
        self._channel_type = channel_type
        self._csi_mode = csi_mode
        
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n, num_bits_per_symbol)
        constellation = Constellation("qam", num_bits_per_symbol)
        self.constellation = constellation
        self._mapper = Mapper(constellation=constellation)
        
        self._channel = AWGN()
        self._demapper = Demapper("app", constellation=constellation)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    def forward(self, batch_size, ebno_db):
        if ebno_db.dim() == 0:
            ebno_db = ebno_db.expand(batch_size)
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate)
        no = expand_to_rank(no, 2)
        
        b = self._binary_source([batch_size, k])
        c = self._encoder(b)
        x = self._mapper(c)
        
        if self._channel_type == 'awgn':
            y = self._channel(x, no)
            no_eff = no
        elif self._channel_type == 'rayleigh':
            y, no_eff = apply_rayleigh_fading(x, no, self._csi_mode, csi_error_var)
        else:
            raise ValueError(f"Unknown channel_type: {self._channel_type}")
        
        llr = self._demapper(y, no_eff)
        b_hat = self._decoder(llr)
        return b, b_hat


# ============================================================
# Training Functions
# ============================================================
def conventional_training(model, ebno_min, ebno_max, num_iter, label=""):
    optimizer = torch.optim.Adam(model.parameters())
    for i in range(num_iter):
        optimizer.zero_grad()
        ebno_db = torch.empty(training_batch_size, device=device).uniform_(ebno_min, ebno_max)
        loss = model(training_batch_size, ebno_db)
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(f'  [{label}] Iter {i}/{num_iter}  BCE: {loss.item():.4f}', end='\r')
    print()
    model.eval()
    optimizer.zero_grad(set_to_none=True)


def rl_based_training(model, ebno_min, ebno_max, num_alt, num_finetune, label=""):
    optimizer_tx = torch.optim.Adam(model.parameters())
    optimizer_rx = torch.optim.Adam(model.parameters())
    
    for i in range(num_alt):
        for _ in range(10):
            optimizer_rx.zero_grad()
            ebno_db = torch.empty(training_batch_size, device=device).uniform_(ebno_min, ebno_max)
            _, rx_loss = model(training_batch_size, ebno_db)
            rx_loss.backward()
            optimizer_rx.step()
        
        optimizer_tx.zero_grad()
        ebno_db = torch.empty(training_batch_size, device=device).uniform_(ebno_min, ebno_max)
        tx_loss, _ = model(training_batch_size, ebno_db, rl_perturbation_var)
        tx_loss.backward()
        optimizer_tx.step()
        
        if i % 500 == 0:
            print(f'  [{label}] Alt Iter {i}/{num_alt}  RX BCE: {rx_loss.item():.4f}', end='\r')
    print()
    
    print(f'  [{label}] Receiver fine-tuning...')
    for i in range(num_finetune):
        optimizer_rx.zero_grad()
        ebno_db = torch.empty(training_batch_size, device=device).uniform_(ebno_min, ebno_max)
        _, rx_loss = model(training_batch_size, ebno_db)
        rx_loss.backward()
        optimizer_rx.step()
        if i % 500 == 0:
            print(f'  [{label}] FT Iter {i}/{num_finetune}  BCE: {rx_loss.item():.4f}', end='\r')
    print()
    model.eval()
    optimizer_tx.zero_grad(set_to_none=True)
    optimizer_rx.zero_grad(set_to_none=True)


def save_weights(model, path):
    state = model.state_dict()
    torch.save(state, path)


def load_weights(model, path):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)


# ============================================================
# Quick Smoke Test (very few iterations)
# ============================================================
def smoke_test():
    print("=" * 60)
    print("SMOKE TEST: Verify code runs correctly")
    print("=" * 60)
    
    # Test AWGN
    print("\n--- AWGN Channel ---")
    model = E2ESystemConventionalTraining(training=True, channel_type='awgn').to(device)
    ebno_db = torch.tensor(6.0, device=device)
    loss = model(4, ebno_db)
    print(f"  AWGN BCE forward pass OK, loss={loss.item():.4f}")
    
    model_rl = E2ESystemRLTraining(training=True, channel_type='awgn').to(device)
    tx_loss, rx_loss = model_rl(4, ebno_db, rl_perturbation_var)
    print(f"  AWGN RL forward pass OK, tx_loss={tx_loss.item():.4f}, rx_loss={rx_loss.item():.4f}")
    
    # Test Rayleigh perfect CSI
    print("\n--- Rayleigh + Perfect CSI ---")
    model_r = E2ESystemConventionalTraining(training=True, channel_type='rayleigh', csi_mode='perfect').to(device)
    ebno_db_r = torch.tensor(20.0, device=device)
    loss = model_r(4, ebno_db_r)
    print(f"  Rayleigh BCE forward pass OK, loss={loss.item():.4f}")
    
    model_rl_r = E2ESystemRLTraining(training=True, channel_type='rayleigh', csi_mode='perfect').to(device)
    tx_loss, rx_loss = model_rl_r(4, ebno_db_r, rl_perturbation_var)
    print(f"  Rayleigh RL forward pass OK, tx_loss={tx_loss.item():.4f}, rx_loss={rx_loss.item():.4f}")
    
    # Test Rayleigh imperfect CSI
    print("\n--- Rayleigh + Imperfect CSI ---")
    model_r_imp = E2ESystemConventionalTraining(training=True, channel_type='rayleigh', csi_mode='imperfect').to(device)
    loss = model_r_imp(4, ebno_db_r)
    print(f"  Rayleigh imperfect CSI BCE forward pass OK, loss={loss.item():.4f}")
    
    model_rl_r_imp = E2ESystemRLTraining(training=True, channel_type='rayleigh', csi_mode='imperfect').to(device)
    tx_loss, rx_loss = model_rl_r_imp(4, ebno_db_r, rl_perturbation_var)
    print(f"  Rayleigh imperfect CSI RL forward pass OK, tx_loss={tx_loss.item():.4f}, rx_loss={rx_loss.item():.4f}")
    
    # Test Baseline
    print("\n--- Baseline ---")
    bl = Baseline(channel_type='awgn').to(device)
    b, b_hat = bl(4, ebno_db)
    print(f"  AWGN Baseline forward pass OK, b.shape={b.shape}")
    
    bl_r = Baseline(channel_type='rayleigh', csi_mode='perfect').to(device)
    b, b_hat = bl_r(4, ebno_db_r)
    print(f"  Rayleigh Baseline forward pass OK, b.shape={b.shape}")
    
    # Test backward pass
    print("\n--- Backward Pass ---")
    model_r = E2ESystemConventionalTraining(training=True, channel_type='rayleigh', csi_mode='perfect').to(device)
    loss = model_r(4, ebno_db_r)
    loss.backward()
    print(f"  Rayleigh BCE backward pass OK")
    
    model_rl_r = E2ESystemRLTraining(training=True, channel_type='rayleigh', csi_mode='perfect').to(device)
    tx_loss, rx_loss = model_rl_r(4, ebno_db_r, rl_perturbation_var)
    (tx_loss + rx_loss).backward()
    print(f"  Rayleigh RL backward pass OK")
    
    # Test sim_ber
    print("\n--- sim_ber Test ---")
    bl_r = Baseline(channel_type='rayleigh', csi_mode='perfect').to(device)
    ebno_test = np.array([15.0, 20.0])
    _, bler = sim_ber(bl_r, ebno_test, batch_size=32, num_target_block_errors=50, 
                      max_mc_iter=50, compile_mode=None)
    print(f"  sim_ber OK, BLER={bler.cpu().numpy()}")
    
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


# ============================================================
# Main Training Pipeline
# ============================================================
def train_all_models():
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)
    
    # 0a. AWGN + BCE
    print("\n[1/5] Training BCE autoencoder on AWGN...")
    t0 = time.time()
    model = E2ESystemConventionalTraining(training=True, channel_type='awgn').to(device)
    conventional_training(model, awgn_ebno_db_min, awgn_ebno_db_max,
                          num_training_iterations_conventional, "BCE-AWGN")
    save_weights(model, model_weights_path_awgn_conv)
    print(f"  Done in {time.time()-t0:.0f}s")
    
    # 0b. AWGN + RL
    print("\n[2/5] Training RL autoencoder on AWGN...")
    t0 = time.time()
    model = E2ESystemRLTraining(training=True, channel_type='awgn').to(device)
    rl_based_training(model, awgn_ebno_db_min, awgn_ebno_db_max,
                      num_training_iterations_rl_alt, num_training_iterations_rl_finetuning,
                      "RL-AWGN")
    save_weights(model, model_weights_path_awgn_rl)
    print(f"  Done in {time.time()-t0:.0f}s")
    
    # 1. Rayleigh + Perfect CSI + BCE
    print("\n[3/5] Training BCE autoencoder on Rayleigh + Perfect CSI...")
    t0 = time.time()
    model = E2ESystemConventionalTraining(training=True, channel_type='rayleigh', csi_mode='perfect').to(device)
    conventional_training(model, rayleigh_ebno_db_min, rayleigh_ebno_db_max,
                          num_training_iterations_conventional, "BCE-Ray-Perfect")
    save_weights(model, model_weights_path_rayleigh_conv)
    print(f"  Done in {time.time()-t0:.0f}s")
    
    # 2. Rayleigh + Perfect CSI + RL
    print("\n[4/5] Training RL autoencoder on Rayleigh + Perfect CSI...")
    t0 = time.time()
    model = E2ESystemRLTraining(training=True, channel_type='rayleigh', csi_mode='perfect').to(device)
    rl_based_training(model, rayleigh_ebno_db_min, rayleigh_ebno_db_max,
                      num_training_iterations_rl_alt, num_training_iterations_rl_finetuning,
                      "RL-Ray-Perfect")
    save_weights(model, model_weights_path_rayleigh_rl)
    print(f"  Done in {time.time()-t0:.0f}s")
    
    # 3. Rayleigh + Imperfect CSI + RL
    print("\n[5/5] Training RL autoencoder on Rayleigh + Imperfect CSI...")
    t0 = time.time()
    model = E2ESystemRLTraining(training=True, channel_type='rayleigh', csi_mode='imperfect').to(device)
    rl_based_training(model, rayleigh_ebno_db_min, rayleigh_ebno_db_max,
                      num_training_iterations_rl_alt, num_training_iterations_rl_finetuning,
                      "RL-Ray-Imperfect")
    save_weights(model, model_weights_path_rayleigh_rl_imperfect)
    print(f"  Done in {time.time()-t0:.0f}s")
    
    print("\nAll models trained!")


# ============================================================
# Evaluation
# ============================================================
def evaluate_all():
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # AWGN evaluation range
    awgn_ebno_dbs = np.arange(awgn_ebno_db_min, awgn_ebno_db_max, 0.5)
    # Rayleigh evaluation range
    rayleigh_ebno_dbs = np.arange(rayleigh_ebno_db_min, rayleigh_ebno_db_max, 0.5)
    
    BLER = {}
    
    with torch.no_grad():
        # --- AWGN (using existing weights if available) ---
        print("\nEvaluating AWGN systems...")
        try:
            model_bl = Baseline(channel_type='awgn').to(device)
            _, bler = sim_ber(model_bl, awgn_ebno_dbs, batch_size=128,
                              num_target_block_errors=200, max_mc_iter=200, compile_mode=None)
            BLER['awgn-baseline'] = bler.cpu().numpy()
            print(f"  AWGN baseline done")
        except Exception as e:
            print(f"  AWGN baseline failed: {e}")
        
        try:
            model_conv = E2ESystemConventionalTraining(training=False, channel_type='awgn').to(device)
            load_weights(model_conv, model_weights_path_awgn_conv)
            _, bler = sim_ber(model_conv, awgn_ebno_dbs, batch_size=128,
                              num_target_block_errors=200, max_mc_iter=200, compile_mode=None)
            BLER['awgn-conv'] = bler.cpu().numpy()
            print(f"  AWGN BCE done")
        except Exception as e:
            print(f"  AWGN BCE skipped (no weights): {e}")
        
        try:
            model_rl = E2ESystemRLTraining(training=False, channel_type='awgn').to(device)
            load_weights(model_rl, model_weights_path_awgn_rl)
            _, bler = sim_ber(model_rl, awgn_ebno_dbs, batch_size=128,
                              num_target_block_errors=200, max_mc_iter=200, compile_mode=None)
            BLER['awgn-rl'] = bler.cpu().numpy()
            print(f"  AWGN RL done")
        except Exception as e:
            print(f"  AWGN RL skipped (no weights): {e}")
        
        # --- Rayleigh + Perfect CSI ---
        print("\nEvaluating Rayleigh + Perfect CSI systems...")
        model_bl_r = Baseline(channel_type='rayleigh', csi_mode='perfect').to(device)
        _, bler = sim_ber(model_bl_r, rayleigh_ebno_dbs, batch_size=128,
                          num_target_block_errors=200, max_mc_iter=200, compile_mode=None)
        BLER['rayleigh-baseline'] = bler.cpu().numpy()
        print(f"  Rayleigh baseline done")
        
        model_conv_r = E2ESystemConventionalTraining(training=False, channel_type='rayleigh', csi_mode='perfect').to(device)
        load_weights(model_conv_r, model_weights_path_rayleigh_conv)
        _, bler = sim_ber(model_conv_r, rayleigh_ebno_dbs, batch_size=128,
                          num_target_block_errors=200, max_mc_iter=200, compile_mode=None)
        BLER['rayleigh-conv-perfect'] = bler.cpu().numpy()
        print(f"  Rayleigh BCE (perfect CSI) done")
        
        model_rl_r = E2ESystemRLTraining(training=False, channel_type='rayleigh', csi_mode='perfect').to(device)
        load_weights(model_rl_r, model_weights_path_rayleigh_rl)
        _, bler = sim_ber(model_rl_r, rayleigh_ebno_dbs, batch_size=128,
                          num_target_block_errors=200, max_mc_iter=200, compile_mode=None)
        BLER['rayleigh-rl-perfect'] = bler.cpu().numpy()
        print(f"  Rayleigh RL (perfect CSI) done")
        
        # --- Rayleigh + Imperfect CSI ---
        print("\nEvaluating Rayleigh + Imperfect CSI systems...")
        # Baseline with imperfect CSI
        model_bl_r_imp = Baseline(channel_type='rayleigh', csi_mode='imperfect').to(device)
        _, bler = sim_ber(model_bl_r_imp, rayleigh_ebno_dbs, batch_size=128,
                          num_target_block_errors=200, max_mc_iter=200, compile_mode=None)
        BLER['rayleigh-baseline-imperfect'] = bler.cpu().numpy()
        print(f"  Rayleigh baseline (imperfect CSI) done")
        
        # BCE trained with perfect CSI, tested with imperfect CSI (model mismatch)
        model_conv_r_mismatch = E2ESystemConventionalTraining(training=False, channel_type='rayleigh', csi_mode='imperfect').to(device)
        load_weights(model_conv_r_mismatch, model_weights_path_rayleigh_conv)  # Perfect CSI weights!
        _, bler = sim_ber(model_conv_r_mismatch, rayleigh_ebno_dbs, batch_size=128,
                          num_target_block_errors=200, max_mc_iter=200, compile_mode=None)
        BLER['rayleigh-conv-mismatch'] = bler.cpu().numpy()
        print(f"  Rayleigh BCE (model mismatch) done")
        
        # RL trained with imperfect CSI, tested with imperfect CSI
        model_rl_r_imp = E2ESystemRLTraining(training=False, channel_type='rayleigh', csi_mode='imperfect').to(device)
        load_weights(model_rl_r_imp, model_weights_path_rayleigh_rl_imperfect)
        _, bler = sim_ber(model_rl_r_imp, rayleigh_ebno_dbs, batch_size=128,
                          num_target_block_errors=200, max_mc_iter=200, compile_mode=None)
        BLER['rayleigh-rl-imperfect'] = bler.cpu().numpy()
        print(f"  Rayleigh RL (imperfect CSI) done")
    
    # Save results
    with open(results_filename, 'wb') as f:
        pickle.dump({
            'awgn_ebno_dbs': awgn_ebno_dbs,
            'rayleigh_ebno_dbs': rayleigh_ebno_dbs,
            'BLER': BLER
        }, f)
    print(f"\nResults saved to {results_filename}")
    return BLER, awgn_ebno_dbs, rayleigh_ebno_dbs


# ============================================================
# Plotting
# ============================================================
def plot_results(BLER, awgn_ebno_dbs, rayleigh_ebno_dbs):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: AWGN vs Rayleigh (Perfect CSI)
    ax = axes[0]
    if 'awgn-baseline' in BLER:
        ax.semilogy(awgn_ebno_dbs, BLER['awgn-baseline'], 'o-', c='C0', label='AWGN Baseline')
    if 'awgn-conv' in BLER:
        ax.semilogy(awgn_ebno_dbs, BLER['awgn-conv'], 'x-.', c='C1', label='AWGN Autoencoder (BCE)')
    if 'awgn-rl' in BLER:
        ax.semilogy(awgn_ebno_dbs, BLER['awgn-rl'], 's--', c='C2', label='AWGN Autoencoder (RL)')
    ax.set_xlabel('Eb/N0 [dB]')
    ax.set_ylabel('BLER')
    ax.set_title('AWGN Channel')
    ax.legend()
    ax.grid(True, which='both')
    ax.set_ylim([1e-4, 1])
    
    # Plot 2: Rayleigh + Perfect CSI
    ax = axes[1]
    if 'rayleigh-baseline' in BLER:
        ax.semilogy(rayleigh_ebno_dbs, BLER['rayleigh-baseline'], 'o-', c='C0', label='Baseline')
    if 'rayleigh-conv-perfect' in BLER:
        ax.semilogy(rayleigh_ebno_dbs, BLER['rayleigh-conv-perfect'], 'x-.', c='C1', label='Autoencoder (BCE)')
    if 'rayleigh-rl-perfect' in BLER:
        ax.semilogy(rayleigh_ebno_dbs, BLER['rayleigh-rl-perfect'], 's--', c='C2', label='Autoencoder (RL)')
    ax.set_xlabel('Eb/N0 [dB]')
    ax.set_ylabel('BLER')
    ax.set_title('Rayleigh Fading + Perfect CSI')
    ax.legend()
    ax.grid(True, which='both')
    ax.set_ylim([1e-4, 1])
    
    # Plot 3: Rayleigh + Imperfect CSI (Model Mismatch Analysis)
    ax = axes[2]
    if 'rayleigh-baseline-imperfect' in BLER:
        ax.semilogy(rayleigh_ebno_dbs, BLER['rayleigh-baseline-imperfect'], 'o-', c='C0', label='Baseline')
    if 'rayleigh-conv-mismatch' in BLER:
        ax.semilogy(rayleigh_ebno_dbs, BLER['rayleigh-conv-mismatch'], 'x-.', c='C1', label='BCE (model mismatch)')
    if 'rayleigh-rl-imperfect' in BLER:
        ax.semilogy(rayleigh_ebno_dbs, BLER['rayleigh-rl-imperfect'], 's--', c='C2', label='RL (trained w/ imperfect CSI)')
    ax.set_xlabel('Eb/N0 [dB]')
    ax.set_ylabel('BLER')
    ax.set_title('Rayleigh Fading + Imperfect CSI')
    ax.legend()
    ax.grid(True, which='both')
    ax.set_ylim([1e-4, 1])
    
    plt.tight_layout()
    plt.savefig('rayleigh_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved to rayleigh_results.png")
    plt.close()
    
    # Constellation plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # BCE constellation
    try:
        model_conv = E2ESystemConventionalTraining(training=True, channel_type='rayleigh', csi_mode='perfect').to(device)
        load_weights(model_conv, model_weights_path_rayleigh_conv)
        pts = torch.complex(model_conv.points_r, model_conv.points_i).detach().cpu().numpy()
        axes[0].scatter(pts.real, pts.imag, s=100)
        axes[0].set_title('BCE (Rayleigh + Perfect CSI)')
        axes[0].set_aspect('equal')
        axes[0].grid(True)
        axes[0].set_xlim([-1.5, 1.5])
        axes[0].set_ylim([-1.5, 1.5])
    except Exception as e:
        axes[0].set_title(f'BCE: {e}')
    
    # RL perfect CSI constellation
    try:
        model_rl = E2ESystemRLTraining(training=True, channel_type='rayleigh', csi_mode='perfect').to(device)
        load_weights(model_rl, model_weights_path_rayleigh_rl)
        pts = torch.complex(model_rl.points_r, model_rl.points_i).detach().cpu().numpy()
        axes[1].scatter(pts.real, pts.imag, s=100)
        axes[1].set_title('RL (Rayleigh + Perfect CSI)')
        axes[1].set_aspect('equal')
        axes[1].grid(True)
        axes[1].set_xlim([-1.5, 1.5])
        axes[1].set_ylim([-1.5, 1.5])
    except Exception as e:
        axes[1].set_title(f'RL: {e}')
    
    # RL imperfect CSI constellation
    try:
        model_rl_imp = E2ESystemRLTraining(training=True, channel_type='rayleigh', csi_mode='imperfect').to(device)
        load_weights(model_rl_imp, model_weights_path_rayleigh_rl_imperfect)
        pts = torch.complex(model_rl_imp.points_r, model_rl_imp.points_i).detach().cpu().numpy()
        axes[2].scatter(pts.real, pts.imag, s=100)
        axes[2].set_title('RL (Rayleigh + Imperfect CSI)')
        axes[2].set_aspect('equal')
        axes[2].grid(True)
        axes[2].set_xlim([-1.5, 1.5])
        axes[2].set_ylim([-1.5, 1.5])
    except Exception as e:
        axes[2].set_title(f'RL imp: {e}')
    
    plt.tight_layout()
    plt.savefig('rayleigh_constellations.png', dpi=150, bbox_inches='tight')
    print("Constellation plot saved to rayleigh_constellations.png")
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        smoke_test()
    elif len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_all_models()
    elif len(sys.argv) > 1 and sys.argv[1] == 'eval':
        BLER, awgn_dbs, rayleigh_dbs = evaluate_all()
        plot_results(BLER, awgn_dbs, rayleigh_dbs)
    elif len(sys.argv) > 1 and sys.argv[1] == 'all':
        smoke_test()
        train_all_models()
        BLER, awgn_dbs, rayleigh_dbs = evaluate_all()
        plot_results(BLER, awgn_dbs, rayleigh_dbs)
    else:
        print("Usage: python rayleigh_autoencoder.py [smoke|train|eval|all]")
        print("  smoke  - Run quick smoke test")
        print("  train  - Train all Rayleigh models")
        print("  eval   - Evaluate and plot results")
        print("  all    - Run smoke test, train, evaluate, and plot")
