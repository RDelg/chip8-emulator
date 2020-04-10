import numpy as np

class Cpu(object):

    _N_REGISTERS = 16
    _MAX_STACK_LEN = 16
    _RAM_BYTES = 4096
    _DISPLAY_SIZE = (64,32)
    _N_KEYS = 16

    def __init__(self):
        # Registers
        # General purpose registers
        self.reg_V = np.zeros(shape=self._N_REGISTERS, dtype=np.uint8)
        # Memory adressing register
        self.reg_VI = np.uint16(0x0000)
        # Special purpose registers
        self.reg_DT = np.uint8(0x00)
        self.reg_ST = np.uint8(0x00)
        # Program counter
        self.reg_PC = np.uint16(0x0000)
        # Stack pointer
        self.reg_SP = np.uint8(0x00)
        # Stack
        self.stack = np.zeros(shape=self._MAX_STACK_LEN, dtype=np.uint16)
        # Memory
        self.memory = np.zeros(shape=(self._RAM_BYTES), dtype=np.uint16)
        self.memory[0:80] = np.array([
                0xF0, 0x90, 0x90, 0x90, 0xF0, # 0
                0x20, 0x60, 0x20, 0x20, 0x70, # 1
                0xF0, 0x10, 0xF0, 0x80, 0xF0, # 2
                0xF0, 0x10, 0xF0, 0x10, 0xF0, # 3
                0x90, 0x90, 0xF0, 0x10, 0x10, # 4
                0xF0, 0x80, 0xF0, 0x10, 0xF0, # 5
                0xF0, 0x80, 0xF0, 0x90, 0xF0, # 6
                0xF0, 0x10, 0x20, 0x40, 0x40, # 7
                0xF0, 0x90, 0xF0, 0x90, 0xF0, # 8
                0xF0, 0x90, 0xF0, 0x10, 0xF0, # 9
                0xF0, 0x90, 0xF0, 0x90, 0x90, # A
                0xE0, 0x90, 0xE0, 0x90, 0xE0, # B
                0xF0, 0x80, 0x80, 0x80, 0xF0, # C
                0xE0, 0x90, 0x90, 0x90, 0xE0, # D
                0xF0, 0x80, 0xF0, 0x80, 0xF0, # E
                0xF0, 0x80, 0xF0, 0x80, 0x80  # F
            ],
            dtype=np.uint8
        )
        self.video_memory = np.zeros(
            shape=self._DISPLAY_SIZE[0]*self._DISPLAY_SIZE[1],
            dtype=np.uint8
        )
        # key states
        self.key_states = np.zeros(shape=self._N_KEYS, dtype=np.uint8)

    def cycle(self):
        instruction = self.memory[self.reg_PC] << 8 ^ self.memory[self.reg_PC+1]
        nnn = instruction & 0x0FFF
        n = instruction & 0x00F
        x = (instruction & 0x0F00) >> 8
        y = (instruction & 0x00F0) >> 4
        kk = instruction & 0x0FF
        # 00E0 - CLS - Clear the display.
        if instruction == 0x00E0: 
            self.video_memory[...] = 0
            self.update_screen()
        # 00EE - RET - Return from a subroutine.
        elif instruction == 0x00EE:
            self.reg_SP -= 1
            self.reg_PC = self.stack[self.reg_SP]
        # 1nnn - JP addr - Jump to location nnn.
        elif instruction & 0xF000 == 0x1000:
            self.reg_PC = nnn - 2
        # 2nnn - CALL addr - Call subroutine at nnn
        elif instruction & 0xF000 == 0x2000:
            self.stack[self.reg_SP] = self.reg_PC
            self.reg_SP += 1
            self.reg_PC = (instruction & 0x0FFF) - 2
        # 3xkk - SE Vx, byte - Skip next instruction if Vx = kk
        elif instruction & 0xF000 == 0x3000:
            if self.reg_V[x] == kk:
                self.reg_PC += 2
        # 4xkk - SNE Vx, byte - Skip next instruction if Vx != kk
        elif instruction & 0xF000 == 0x4000:
            if self.reg_V[x] != kk:
                self.reg_PC += 2
        # 5xy0 - SE Vx, Vy - Skip next instruction if Vx = Vy
        elif instruction & 0xF000 == 0x5000:
            if self.reg_V[x] == self.reg_V[y]:
                self.reg_PC += 2 
        # 6xkk - LD Vx, byte - Set Vx = kk.
        elif instruction & 0xF000 == 0x6000:
            self.reg_V[x] = kk
        # 7xkk - ADD Vx, byte - Set Vx = Vx + kk
        elif instruction & 0xF000 == 0x7000:
            self.reg_V[x] += kk
        elif instruction & 0xF000 == 0x8000:
            # 8xy0 - LD Vx, Vy - Set Vx = Vy
            if n = 0x0:
                self.reg_V[x] = self.reg_V[y]
            # 8xy1 - OR Vx, Vy - Set Vx = Vx OR Vy
            elif n = 0x1:
                self.reg_V[x] |= self.reg_V[y]
            # 8xy2 - AND Vx, Vy - Set Vx = Vx AND Vy
            elif n = 0x2:
                self.reg_V[x] &= self.reg_V[y]
            # 8xy3 - XOR Vx, Vy - Set Vx = Vx XOR Vy
            elif n = 0x3:
                self.reg_V[x] ^= self.reg_V[y]
            # 8xy4 - ADD Vx, Vy - Set Vx = Vx + Vy, set VF = carry
            elif n = 0x4:
                s = np.uint16(self.reg_V[x]) + np.uint16(self.reg_V[y])
                self.reg_V[0xF] = (s) > np.uint16(0xFF);
                self.reg_V[x] = np.uint8(s)
            # 8xy5 - SUB Vx, Vy - Set Vx = Vx - Vy, set VF = NOT borrow
            elif n = 0x5:
                self.reg_V[0xF] = self.reg_V[x] > self.reg_V[y]
                self.reg_V[x] -= self.reg_V[y]
            # 8xy6 - SHR Vx {, Vy} - Set Vx = Vx SHR 1
            elif n = 0x6:
                self.reg_V[0xF] = (self.reg_V[X] & 0x01);
                self.reg_V[x] >>= 1;
            # 8xy7 - SUBN Vx, Vy - Set Vx = Vy - Vx, set VF = NOT borrow
            elif n = 0x7:
                self.reg_V[0xF] = self.reg_V[y] > self.reg_V[x]
                self.reg_V[x] = self.reg_V[y] - self.reg_V[x]
            # 8xyE - SHL Vx {, Vy} - Set Vx = Vx SHL 1
            elif n = 0xE:
                self.reg_V[0xF] = (self.reg_V[x] >> 7);
                self.reg_V[x] <<= 1;
        # 9xy0 - SNE Vx, Vy - Skip next instruction if Vx != Vy
        elif instruction & 0xF000 == 0x9000:
            if self.reg_V[x] != self.reg_V[y]:
                self.reg_PC += 2 
        # Annn - LD I, addr - Set I = nnn
        elif instruction & 0xF000 == 0xA000:
            self.reg_VI = nnn
        # Bnnn - JP V0, addr - Jump to location nnn + V0.
        elif instruction & 0xF000 == 0xB000:
            self.reg_PC = nnn + self.reg_V[0x0] - 1
        # Cxkk - RND Vx, byte - Set Vx = random byte AND kk
        elif instruction & 0xF000 == 0xC000:
            self.reg_V[x] = np.random.randint(0, 255, dtype=np.uint8) & kk
        # Dxyn - DRW Vx, Vy, nibble - Display n-byte sprite starting at memory location I at (Vx, Vy), set VF = collision
        elif instruction & 0xF000 == 0xD000
            self.draw_sprite(self.reg_V[x], self.reg_V[y], n)
            self.update_screen()
        elif instruction & 0xF000 == 0xE000:
            # Ex9E - SKP Vx - Skip next instruction if key with the value of Vx is pressed
            if n = 0xE:
                if self.key_states[self.reg_V[x]]:
                    self.reg_PC += 2
            # ExA1 - SKNP Vx - Skip next instruction if key with the value of Vx is not pressed
            elif n = 0x1:
                if not self.key_states[self.reg_V[x]]:
                    self.reg_PC += 2
        elif instruction & 0xF000 == 0xF000:
            # Fx07 - LD Vx, DT - Set Vx = delay timer value
            if n = 0x07:
                self.reg_V[x] = self.reg_DT
            # Fx0A - LD Vx, K - Wait for a key press, store the value of the key in Vx
            elif n = 0x0A:
                self.reg_V[x] = self.wait_for_key()
            # Fx15 - LD DT, Vx - Set delay timer = Vx
            elif n = 0x15:
                self.reg_DT = self.reg_V[x]
            # Fx18 - LD ST, Vx - Set sound timer = Vx
            elif n = 0x18:
                self.reg_ST = self.reg_V[x]
            # Fx1E - ADD I, Vx - Set I = I + Vx
            elif n = 0x1E:
                self.reg_VI += self.reg_V[x]
            # Fx29 - LD F, Vx - Set I = location of sprite for digit Vx
            elif n = 0x29:
                self.reg_VI = self.reg_V[x] * 5
            # Fx33 - LD B, Vx - Store BCD representation of Vx in memory locations I, I+1, and I+2
            elif n = 0x33:
                self.memory[self.reg_VI]     = self.reg_V[x] / 100;
                self.memory[self.reg_VI + 1] = self.reg_V[x] % 100 / 10;
                self.memory[self.reg_VI + 2] = self.reg_V[x] % 10;
            # Fx55 - LD [I], Vx - Store registers V0 through Vx in memory starting at location I
            elif n = 0x55:
                 for i in range(self._N_REGISTERS):
                    self.memory[self.reg_VI + i] = self.reg_V[i]
            # Fx65 - LD Vx, [I] - Read registers V0 through Vx from memory starting at location I
            elif n = 0x65:
                for i in range(self._N_REGISTERS):
                     self.reg_V[i] = self.memory[self.reg_VI + i]
        self.reg_PC += 2


    def draw_sprite(self):
        raise NotImplementedError

    def update_screen(self):
        raise NotImplementedError