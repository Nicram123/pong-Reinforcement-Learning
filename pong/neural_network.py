
# neural_network.py
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

class PongEnv:
    def __init__(self, width=800, height=600, paddle_speed=20, ball_speed=10):
        self.width = width
        self.height = height
        self.paddle_speed = paddle_speed
        self.ball_speed = ball_speed
        self.reset()

    def reset(self):
        # Pozycje paletek i piłki
        self.l_paddle_y = 0
        self.r_paddle_y = 0
        self.ball_x = 0
        self.ball_y = 0
        # Kierunki ruchu piłki (losowe na starcie)
        self.ball_dx = np.random.choice([-self.ball_speed, self.ball_speed])
        self.ball_dy = np.random.choice([-self.ball_speed, self.ball_speed])
        # Nowe cechy
        diff_r = self.ball_y - self.r_paddle_y
        diff_l = self.ball_y - self.l_paddle_y
        
        # Normalizacja
        norm = lambda v, m: v / m
        state = np.array([
            norm(self.ball_x, self.width/2),
            norm(self.ball_y, self.height/2),
            norm(self.ball_dx, self.ball_speed),
            norm(self.ball_dy, self.ball_speed),
            norm(self.r_paddle_y, self.height/2),
            norm(self.l_paddle_y, self.height/2),
            norm(diff_r, self.height/2),
            norm(diff_l, self.height/2)
            ], dtype=np.float32)
        return state

    def step(self, action):
        """
        Akcje:
        0 = stay
        1 = up
        2 = down
        """
        # --- sterowanie paletką AI (prawa) ---
        if action == 1: # wybrał up 
            self.r_paddle_y += self.paddle_speed
        elif action == 2: # wybrał down 
            self.r_paddle_y -= self.paddle_speed
            
        # ograniczenie paletki do ekranu (żeby nie wychodziła poza ekran)
        half_height = self.height / 2 - 50
        self.r_paddle_y = np.clip(self.r_paddle_y, -half_height, half_height)  
          
        # --- ruch piłki ---
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # --- kolizja ze ścianą góra/dół ---
        if self.ball_y > self.height/2 - 10 or self.ball_y < -self.height/2 + 10:
            self.ball_dy *= -1
            #self.ball_x = self.width/2 - 51
        reward = 0
        done = False
        
        # --- kolizja z paletką AI ---
        if (self.ball_x > (self.width/2 - 60)) and (abs(self.ball_y - self.r_paddle_y) < 60):
            self.ball_dx *= -1
            #self.ball_x = self.width/2 + 51
            self.ball_x = self.width/2 - 60
            reward += 1 # 0.1  # nagroda za odbicie
            print("AI hit the ball")
            
        #move = np.random.choice([-self.paddle_speed, 0, self.paddle_speed])
        #self.l_paddle_y += move
        
        # bot podąża za piłką z lekkim opóźnieniem
        #if self.ball_y > self.l_paddle_y + 10:
        #    self.l_paddle_y += self.paddle_speed
        #elif self.ball_y < self.l_paddle_y - 10:
        #    self.l_paddle_y -= self.paddle_speed
        if np.random.rand() < 0.4:  # 70% szansy że w ogóle ruszy
            if self.ball_y > self.l_paddle_y + 20:
                self.l_paddle_y += self.paddle_speed
            elif self.ball_y < self.l_paddle_y - 20:
                self.l_paddle_y -= self.paddle_speed

        # clipsowanie lewej paletki 
        self.l_paddle_y = np.clip(self.l_paddle_y, -half_height, half_height)
        
        # --- kolizja z paletką przeciwnika (losowy bot) ---
        if (self.ball_x < -(self.width/2 - 60)) and (abs(self.ball_y - self.l_paddle_y) < 60):
            self.ball_dx *= -1
            self.ball_x = -(self.width/2 - 60)
            
        # --- punkt dla którejś strony ---
        if self.ball_x > self.width/2:
            print("me pointed")
            reward -= 5 # -1
            done = True
            
        elif self.ball_x < -self.width/2:
            print("AI pointed")
            reward += 5 # 1
            done = True
            
        reward += 0.01
        
        diff_r = self.ball_y - self.r_paddle_y
        diff_l = self.ball_y - self.l_paddle_y
        
        # Normalizacja
        norm = lambda v, m: v / m
        
        # --- nowy stan ---
        state = np.array([
            norm(self.ball_x, self.width/2),
            norm(self.ball_y, self.height/2),
            norm(self.ball_dx, self.ball_speed),
            norm(self.ball_dy, self.ball_speed),
            norm(self.r_paddle_y, self.height/2),
            norm(self.l_paddle_y, self.height/2),
            norm(diff_r, self.height/2),
            norm(diff_l, self.height/2)
            ], dtype=np.float32)
        return state, reward, done

# --- Sieć Q-learning ---
def build_q_network(state_size=8, num_actions=3):
    model = Sequential([
        Input((state_size,)),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_actions)
    ])
    return model


