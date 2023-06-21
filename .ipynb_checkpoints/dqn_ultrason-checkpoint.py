import tensorflow as tf
import numpy as np
import RPi.GPIO as GPIO
import time
import pickle
import cv2


#Definition of  motor pins 
IN1 = 20
IN2 = 21
IN3 = 19
IN4 = 26
ENA = 16
ENB = 13

#Definition of  key
key = 8

#Definition of  ultrasonic module pins
EchoPin = 0
TrigPin = 1

EchoPinB = 12
TrigPinB = 17

#Definition of RGB module pins
LED_R = 22
LED_G = 27
LED_B = 24

#Definition of servo pin
ServoPin = 23

#Definition of infrared obstacle avoidance module pins
#AvoidSensorLeft = 12
#AvoidSensorRight = 17

#Set the GPIO port to BCM encoding mode
GPIO.setmode(GPIO.BCM)

#Ignore warning information
GPIO.setwarnings(False)

#Buzzer pin
buzzer_pin = 8

#Motor pins are initialized into output mode
#Key pin is initialized into input mode
#Ultrasonic pin,RGB pin,servo pin initialization
#infrared obstacle avoidance module pin
def ORB_detector(new_image, image_template):
    # Function that compares input image to template
    # It then returns the number of ORB matches between them
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB_create(1000, 1.2)

    # Detect keypoints of original image
    (kp1, des1) = orb.detectAndCompute(image1, None)

    # Detect keypoints of rotated image
    (kp2, des2) = orb.detectAndCompute(image_template, None)

    # Create matcher 
    # Note we're no longer using Flannbased matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Do matching
    matches = bf.match(des1,des2)

    # Sort the matches based on distance.  Least distance
    # is better
    matches = sorted(matches, key=lambda val: val.distance)
    return len(matches)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    


def init():
    global pwm_ENA
    global pwm_ENB
    global pwm_servo
    GPIO.setup(ENA,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN1,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN2,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(ENB,GPIO.OUT,initial=GPIO.HIGH)
    GPIO.setup(IN3,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(IN4,GPIO.OUT,initial=GPIO.LOW)
    GPIO.setup(key,GPIO.IN)
    GPIO.setup(EchoPin,GPIO.IN)
    GPIO.setup(TrigPin,GPIO.OUT)
    GPIO.setup(EchoPinB,GPIO.IN)
    GPIO.setup(TrigPinB,GPIO.OUT)
    GPIO.setup(LED_R, GPIO.OUT)
    GPIO.setup(LED_G, GPIO.OUT)
    GPIO.setup(LED_B, GPIO.OUT)
    GPIO.setup(ServoPin, GPIO.OUT)
    #GPIO.setup(AvoidSensorLeft,GPIO.IN)
    #GPIO.setup(AvoidSensorRight,GPIO.IN)
    #GPIO.setup(buzzer_pin, GPIO.OUT)
    #Set the PWM pin and frequency is 2000hz
    pwm_ENA = GPIO.PWM(ENA, 2000)
    pwm_ENB = GPIO.PWM(ENB, 2000)
    pwm_ENA.start(0)
    pwm_ENB.start(0)

    pwm_servo = GPIO.PWM(ServoPin, 50)
    pwm_servo.start(0)


# Настройка пина пищалки на выход


# Функция для проигрывания звука на пищалке
# def play_buzzer(duration):
#     GPIO.output(buzzer_pin, GPIO.HIGH)  # Включить пищалку
#     time.sleep(duration)  # Установить продолжительность звучания
#     GPIO.output(buzzer_pin, GPIO.LOW)  # Выключить пищалку

    
#advance
def run(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)

#back
def back(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)
        
#turn left
def left(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)

#trun right 
def right(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)

#turn left in place
def spin_left(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)

#turn right in place
def spin_right(leftspeed, rightspeed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(leftspeed)
    pwm_ENB.ChangeDutyCycle(rightspeed)

#brake
def brake():
   GPIO.output(IN1, GPIO.LOW)
   GPIO.output(IN2, GPIO.LOW)
   GPIO.output(IN3, GPIO.LOW)
   GPIO.output(IN4, GPIO.LOW)

def Distance():
    GPIO.output(TrigPin,GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin,GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin,GPIO.LOW)

    t3 = time.time()
    while not GPIO.input(EchoPin):
        t4 = time.time()
        if (t4 - t3) > 0.03 :
            return -1
    t1 = time.time()
    while GPIO.input(EchoPin):
        t5 = time.time()
        if(t5 - t1) > 0.03 :
            return -1

    t2 = time.time()
    #print "distance is %d " % (((t2 - t1)* 340 / 2) * 100)
    time.sleep(0.01)
    return ((t2 - t1)* 340 / 2) * 100

def DistanceBehind():
    GPIO.output(TrigPinB,GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPinB,GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPinB,GPIO.LOW)

    t3 = time.time()
    while not GPIO.input(EchoPinB):
        t4 = time.time()
        if (t4 - t3) > 0.03 :
            return -1
    t1 = time.time()
    while GPIO.input(EchoPinB):
        t5 = time.time()
        if(t5 - t1) > 0.03 :
            return -1

    t2 = time.time()
    #print "distance is %d " % (((t2 - t1)* 340 / 2) * 100)
    time.sleep(0.01)
    return ((t2 - t1)* 340 / 2) * 100


def Distance_test():
    num = 0
    ultrasonic = []
    while num < 5:
            distance = Distance()
            while int(distance) == -1 :
                distance = Distance()
                print("Tdistance is %f"%(distance) )
            while (int(distance) >= 500 or int(distance) == 0) :
                distance = Distance()
                print("Edistance is %f"%(distance) )
            ultrasonic.append(distance)
            num = num + 1
            time.sleep(0.01)
    print(ultrasonic)
    distance = (ultrasonic[1] + ultrasonic[2] + ultrasonic[3])/3
    print("distance is %f"%(distance) )
    distance_list = np.array(ultrasonic)
    return distance



def Distance_testBehind():
    num = 0
    ultrasonic = []
    while num < 5:
            distance = DistanceBehind()
            while int(distance) == -1 :
                distance = DistanceBehind()
                print("Tdistance BEHIND is %f"%(distance) )
            while (int(distance) >= 500 or int(distance) == 0) :
                distance = DistanceBehind()
                print("Edistance BEHIND is %f"%(distance) )
            ultrasonic.append(distance)
            num = num + 1
            time.sleep(0.01)
    print('BEHIND ', ultrasonic)
    distance = (ultrasonic[1] + ultrasonic[2] + ultrasonic[3])/3
    print("distance BEHIND is %f"%(distance) )
    distance_list = np.array(ultrasonic)
    return distance



# Define the DQN model using TensorFlow
class DQN:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = tf.keras.Sequential([
            #tf.keras.layers.Flatten(input_shape=self.state_dim,),
            tf.keras.layers.Dense(32, activation='relu', input_dim=self.state_dim,),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            print(f"{bcolors.WARNING}ВЫБРАНО СЛУЧАЙНОЕ ЗНАЧЕНИЕ №№!№!№!№!№!№!№!№!№!№!№!№!№!№!№!№!№!№{bcolors.WARNING}")
            return np.random.randint(self.action_dim)
        print(f"{bcolors.WARNING}ЗНАЧЕНИЕ ИЗ ВЕСОВ -------------------------------------------------{bcolors.WARNING}")
        return np.argmax(self.model.predict(state))
#     def choose_action(self, state):
#         if np.random.rand() <= self.epsilon:
#             return np.random.randint(self.action_dim)
#         flattened_state = state.flatten()  # Преобразование состояния в одномерный массив
#         return np.argmax(self.model.predict(flattened_state[np.newaxis, :])[0])


    def train(self, state, action, reward, next_state, done, verbose=True):
#         target = self.model.predict(state)
        target = self.model.predict(state)
        if done:
            target[0][action] = reward
        else:
            q_future = np.max(self.target_model.predict(next_state)[0])
            target[0][action] = reward + self.gamma * q_future
        
        self.model.fit(state, target, epochs=1, verbose=verbose)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        self.target_model = self.build_model()
        self.update_target_model()


init()
# Define the DQN algorithm functions
def preprocess_state(state):
    # Preprocess the state here if necessary
    return state

def get_state(image):
    # Read the sensor inputs and return the current state
    distance = Distance_test()
    distance_b = Distance_testBehind()
    state = np.array([distance, distance_b]).reshape(1, 2)
    image = cv2.resize(image, (320, 240))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    a = []
    a.append(state)
    a.append(image)
    return a

def take_action(action):
    # Take the action based on the selected action index
    if action == 1:
        run(30,30)
        print('ВПЕРЕД')
    elif action == 2:
        left(30,30)
        print('ВЛЕВО')
    elif action == 3:
        back(30,30)
        print('НАЗАД')
    elif action == 4:
        right(30,30)
        print('ВПРАВО')

def is_obstacle_detected(distance, distance_b, detected):
    global done
    # Check if an obstacle is detected
    if detected:
        brake()
        print('OBJECT NAIDEN !!!!!!!!!!!!!!!!!!!!!!!')
        #play_buzzer(0.1)
        done = True
        return False
    
    
    if distance < 15:
        print('OBSTACLE DETECTED FORWARD')
        #brake()
        spin_right(15,15)
        return True
    if distance_b < 15:
        print('OBSTACLE DETECTED BEHIND')
        #brake()
        run(15,15)
        return True


def get_reward(obstacle_detected):
    if obstacle_detected:
        
        print(f"{bcolors.FAIL}REWARD -3{bcolors.FAIL}")
        return -3
    
    print(f"{bcolors.OKGREEN}REWARD +1{bcolors.OKGREEN}")
    return 1

# Initialize the DQN algorithm
state_dim = 2  # Specify the dimension of the state
action_dim = 4  # Specify the number of possible actions
dqn = DQN(state_dim, action_dim)
#dqn.load_model('model_weights.h5')
# Start the main loop for obstacle avoidance

cap = cv2.VideoCapture(0)
image_template = cv2.imread('simple.png', 0)

try:
    while True:
        ret, frame = cap.read()
        # Get the current state
        state = get_state(frame)
        #state = preprocess_state(state)

        # Choose an action using the DQN algorithm
        action = dqn.choose_action(state)

        # Take the chosen action
        take_action(action)
        
        
        ret, frame = cap.read()
        
        
        height, width = frame.shape[:2]
        
        top_left_x = int(width / 3)
        top_left_y = int((height / 2) + (height / 4))
        bottom_right_x = int((width / 3) * 2)
        bottom_right_y = int((height / 2) - (height / 4))
        
        cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]
        
        matches = ORB_detector(cropped, image_template)
        
        detected = False
        
        if matches > 200:
            detected = True
        obstacle_detected = is_obstacle_detected(state[0][0], state[0][1],detected)
        
    
        # Check if an obstacle is detected
        

        # Get the reward based on the obstacle detection
        reward = get_reward(obstacle_detected)

        # Get the next state
        next_state = get_state()
        next_state = preprocess_state(next_state)

        # Check if the episode is done (optional, depends on your setup)
        done = False
        
        
        #ndarray = np.array([state, detected])
        # Train the DQN model with the experience tuple
        dqn.train(state, action, reward, next_state, done, verbose=2) #####################

        # Update the target DQN model periodically
        #if episode % update_target_frequency == 0:
        dqn.update_target_model()

        # Sleep for a short duration to control the loop rate
        time.sleep(0.1)
        

except KeyboardInterrupt:
    pass
finally:
    print(f"{bcolors.UNDERLINE}MODEL SAVED{bcolors.UNDERLINE}")
    dqn.save_model('model_weights.h5')



cap.release()
cv2.destroyAllWindows()
pwm_ENA.stop()
pwm_ENB.stop()
GPIO.cleanup()

