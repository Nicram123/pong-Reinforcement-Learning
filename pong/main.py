from turtle import Screen, Turtle
from paddle import Paddle
from ball import Ball
import time
from scoreboard import Scoreboard
import socket
import pickle
import _tkinter
from tensorflow.keras.models import load_model
import numpy as np

screen = Screen()
screen.bgcolor('black')
screen.setup(width=800, height=600)
#screen.title('Pong')
screen.tracer(0) # wylaczenie animacji 

HEIGHT = 600 

FORMAT = 'utf-8'
HEADER = 70

r_paddle = Paddle((350, 0))
l_paddle = Paddle((-350, 0))
ball = Ball()
scoreboard = Scoreboard()


screen.listen() # nasluchuje na wcisniete klawisze 
screen.onkey(r_paddle.go_up, 'Up') # funkcja wywolana po wcisnieciu gornej strzalki 
screen.onkey(r_paddle.go_down, 'Down') # 


screen.listen() # nasluchuje na wcisniete klawisze 
screen.onkey(l_paddle.go_up, 'w') # funkcja wywolana po wcisnieciu gornej strzalki 
screen.onkey(l_paddle.go_down, 's') # 


    
def update_game_state(state):
    ball.setx(state['ball'][0])
    ball.sety(state['ball'][1])
    l_paddle.setx(state['l_paddle'][0])
    l_paddle.sety(state['l_paddle'][1])
    r_paddle.setx(state['r_paddle'][0])
    r_paddle.sety(state['r_paddle'][1])
    scoreboard.l_score = state['scores'][0]
    scoreboard.r_score = state['scores'][1]
    #screen = state['screen']
    scoreboard.update_scoreboard()
    
    
def clamp_paddle(paddle):
    max_y = 250
    min_y = -250
    if paddle.ycor() > max_y:
        paddle.sety(max_y)
    elif paddle.ycor() < min_y:
        paddle.sety(min_y)

# zawsze gdy wylaczamy animacje trzeba updatowac po nic nie bedzie widoczne 
try:
  game_is_on = True
  iteration = 0
  
  
  #q_network = load_model("pong_ai_ep300.keras", compile=False)
  q_network = load_model("models/pong_ai_ep2200.keras", compile=False) # na 800 i 1600 gra sie super 
  while game_is_on:
    time.sleep(0.1)
    #ball.move_speed
    screen.update()
    # -- new code here --
    ball.move() 
    
    state = np.array(
      [
    ball.xcor() / 400,
    ball.ycor() / 300,
    ball.x_move / 10,
    ball.y_move / 10,
    r_paddle.ycor() / 300,
    l_paddle.ycor() / 300,
    (ball.ycor() - r_paddle.ycor()) / 300,
    (ball.ycor() - l_paddle.ycor()) / 300
   ])
    
    state = np.expand_dims(state, axis=0)  # (1,6)
    q_values = q_network(state)
    action = np.argmax(q_values[0])
    if action == 1:
        r_paddle.sety(r_paddle.ycor() + 20)
    elif action == 2:
        r_paddle.sety(r_paddle.ycor() - 20)
    # -- end of the new code 
    
    
    
    clamp_paddle(r_paddle)  # <-- ograniczenie ruchu AI
    # --- gracz kontroluje lewą paletkę ---
    clamp_paddle(l_paddle)  # <-- ograniczenie ruchu gracza
    
    # Detect collision with wall 
    if ball.ycor() > 280 or ball.ycor() < -280:
      # needs to bounce
      ball.bounce_y()
      
    # Detect collision with paddle
    if ball.distance(r_paddle) < 50 and ball.xcor() > 320 or ball.distance(l_paddle) < 50 and ball.xcor() < -320:
      ball.bounce_x()
      
    # Detect R paddle missed 
    if ball.xcor() > 380:
      ball.reset_position()
      scoreboard.l_point()
    
    # Detect L paddle missed
    if ball.xcor() < -380:
      ball.reset_position()
      scoreboard.r_point()
      
    game_state = {
          'ball': (ball.xcor(), ball.ycor()),
          'l_paddle': (l_paddle.xcor(), l_paddle.ycor()),
          'r_paddle': (r_paddle.xcor(), r_paddle.ycor()),
          'scores': (scoreboard.l_score, scoreboard.r_score),
          
      }
    update_game_state(game_state)
        
    time.sleep(0.01) 
except _tkinter.TclError as e:
    print(f"Błąd: {e}. Ekran turtle został zamknięty.")
finally:
    screen.bye()

screen.exitonclick() 