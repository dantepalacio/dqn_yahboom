import tensorflow as tf
import cv2
import numpy as np
import rpio
import camera as cam
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model

# Определение архитектуры модели
# class DQNNetwork(tf.keras.Model):
#     def __init__(self, num_actions):
#         super(DQNNetwork, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')
#         self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')
#         self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')
#         self.flatten = tf.keras.layers.Flatten()
#         self.fc1 = tf.keras.layers.Dense(512, activation='relu')
#         self.fc2 = tf.keras.layers.Dense(num_actions)

#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         q_values = self.fc2(x)
#         return q_values

class DQNNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(DQNNetwork, self).__init__()
        self.conv1 = Conv2D(32, kernel_size=3, activation='relu')
        self.conv2 = Conv2D(64, kernel_size=3, activation='relu')
        self.flatten = Flatten()
        self.dense = Dense(512, activation='relu')
        self.concat = Concatenate(axis=-1)
        self.output_layer = Dense(num_actions)

    def call(self, inputs):
        image, ultrasonic_data = inputs

        x = self.conv1(image)
        x = self.conv2(x)
        x = self.flatten(x)

        # Reshape ultrasonic_data to match the shape of the flattened conv2d output
        ultrasonic_data = tf.reshape(ultrasonic_data, shape=(-1, 2))

        x = self.concat([x, ultrasonic_data])
        x = self.dense(x)
        output = self.output_layer(x)
        
        return output

    
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
    
# Параметры обучения
epsilon_initial = 1.0
epsilon_final = 0.01
epsilon_decay_steps = 10000
replay_memory_capacity = 10000
batch_size = 32
target_network_update_frequency = 1000
discount_factor = 0.99
learning_rate = 0.001

# Создание и компиляция модели

# Создание модели
num_actions = 3  # Замените на количество доступных действий в вашей задаче
model = DQNNetwork(num_actions)
target_model = DQNNetwork(num_actions)
optimizer = tf.keras.optimizers.Adam(learning_rate)
mse_loss = tf.keras.losses.MeanSquaredError()

# input_image = Input(shape=(240, 320, 1))
# input_ultrasonic_data = Input(shape=(2,))


# output = model([input_image, input_ultrasonic_data])


# full_model = Model(inputs=[input_image, input_ultrasonic_data], outputs=output)
# full_model.compile(optimizer='adam', loss='mse')
# Загрузка предыдущего обучения (если есть)

# Определение стратегии разведения
epsilon = epsilon_initial
def epsilon_greedy_policy(state, epsilon=epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    else:
        q_values = model.predict(state[np.newaxis])
        return np.argmax(q_values[0])

# Создание реплей-буфера
replay_memory = []

# Функция для получения состояния агента
# def get_state(ultrasonic_data, image):
#     # Обработка данных с ультразвуковых датчиков
#     ultrasonic_data = np.expand_dims(ultrasonic_data, axis=0)

#     # Обработка изображения с помощью OpenCV
#     image = cv2.resize(image, (320, 240))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     image = image.flatten()
#     ultrasonic_data = ultrasonic_data.flatten()

#     # Объединение данных в состояние
#     state = np.concatenate((ultrasonic_data, image), axis=-1)
#     return state
def get_state(ultrasonic_data, image):
    # Обработка данных с ультразвуковых датчиков
    #ultrasonic_data = np.expand_dims(ultrasonic_data, axis=0)


    # Обработка изображения с помощью OpenCV
    image = cv2.resize(image, (320, 240))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Объединение данных в состояние
    state = np.concatenate((ultrasonic_data, image), axis=-1)

    #state = np.array([ultrasonic_data, image])
    return state

# Функция для выполнения одного шага обучения
def train_step(replay_memory, model, target_model, optimizer, discount_factor, batch_size):
    # Выбор мини-пакета из реплей-буфера
    batch_indices = np.random.choice(len(replay_memory), size=batch_size, replace=False)
    batch = [replay_memory[index] for index in batch_indices]
    states, actions, rewards, next_states, dones = zip(*batch)

    # Преобразование данных пакета в формат, совместимый с TensorFlow
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.float32)

    # Вычисление Q-значений для текущих состояний и следующих состояний
    q_values = model.predict(states)
    next_q_values = target_model.predict(next_states)

    # Вычисление целевых Q-значений
    targets = np.copy(q_values)
    batch_indices = np.arange(batch_size, dtype=np.int32)
    targets[batch_indices, actions] = rewards + discount_factor * np.max(next_q_values, axis=1) * (1 - dones)

    # Обновление модели
    with tf.GradientTape() as tape:
        q_values = model(states)
        loss_value = mse_loss(targets, q_values)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_value

def take_action(action):
    # Take the action based on the selected action index
    if action == 0:
        rpio.run(30,30)
        print('ВПЕРЕД')
    elif action == 1:
        rpio.left(30,30)
        print('ВЛЕВО')
    elif action == 2:
        rpio.back(30,30)
        print('НАЗАД')
    elif action == 3:
        rpio.right(30,30)
        print('ВПРАВО')
        
def is_obstacle_detected(distance, distance_b):
    if distance < 15:
        print('OBSTACLE DETECTED FORWARD')
        #brake()
        rpio.spin_right(15,15)
        return True
    if distance_b < 15:
        print('OBSTACLE DETECTED BEHIND')
        #brake()
        rpio.run(15,15)
        return True


def get_reward(obstacle_detected):
    if obstacle_detected:
        
        print(f"{bcolors.FAIL}REWARD -2{bcolors.FAIL}")
        return -2
    
    print(f"{bcolors.OKGREEN}REWARD +1{bcolors.OKGREEN}")
    return 1

def done_check(cap):
    global done
    done = False
    ret, frame = cap.read()

    # Get height and width of webcam frame
    height, width = frame.shape[:2]

    # Define ROI Box Dimensions (Note some of these things should be outside the loop)
    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))

    # Draw rectangular window for our region of interest
    cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 3)

    # Crop window of observation we defined above
    cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]

    # Flip frame orientation horizontally
    frame = cv2.flip(frame,1)

    # Get number of ORB matches 
    matches = cam.ORB_detector(cropped, image_template)

    # Display status string showing the current no. of matches 
    output_string = "# of Matches = " + str(matches)
    cv2.putText(frame, output_string, (50,450), cv2.FONT_HERSHEY_COMPLEX, 1, (250,0,0), 2)

    # Our threshold to indicate object deteciton
    # For new images or lightening conditions you may need to experiment a bit 
    # Note: The ORB detector to get the top 1000 matches, 350 is essentially a min 35% match
    threshold = 200

    # If matches exceed our threshold then object has been detected
    if matches > threshold:
        cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
        cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)
        done = True
    return done

# Основной цикл обучения
rpio.init()
fps = 0

cap = cv2.VideoCapture(0)
#cam.translation(cap)
# Load our image template, this is our reference image
image_template = cv2.imread('simple.png', 0) 

num_episodes = 1000  # Замените на необходимое количество эпизодов
for episode in range(num_episodes):
    
    ret, frame = cap.read()

    # Проверка успешности считывания кадра
    if not ret:
        break
    # Инициализация эпизода
    episode_loss = 0.0
    episode_reward = 0.0
    episode_steps = 0

    # Получение начального состояния агента
    distance_front = rpio.Distance_test()
    distance_back = rpio.Distance_testBehind()
    ultrasonic_data = np.array([distance_front, distance_back])
    #image = frame.reshape(-1, 3) 

    state = get_state(ultrasonic_data, frame)

    while True:
        # Выбор действия согласно стратегии разведения
        action = epsilon_greedy_policy(state)
        
        take_action(action)

        # Выполнение выбранного действия и получение нового состояния и награды
        distance_front = rpio.Distance_test()
        distance_back = rpio.Distance_testBehind()
        next_ultrasonic_data = np.array([distance_front, distance_back])
        next_image = frame
        next_state = get_state(next_ultrasonic_data, next_image)
        is_obstacle_detected(distance_front, distance_back)
        reward = get_reward(is_obstacle_detected(distance_front, distance_back))
        
        done = done_check(cap)

        # Добавление перехода в реплей-буфер
        replay_memory.append((state, action, reward, next_state, done))
        if len(replay_memory) > replay_memory_capacity:
            replay_memory.pop(0)

        # Обновление состояния
        state = next_state

        # Обновление целевой модели
        if episode_steps % target_network_update_frequency == 0:
            target_model.set_weights(model.get_weights())

        # Обучение модели
        if len(replay_memory) > batch_size:
            loss = train_step(replay_memory, model, target_model, optimizer, discount_factor, batch_size)
            episode_loss += loss

        # Обновление переменных
        episode_reward += reward
        episode_steps += 1

        if done:
            break

    # Уменьшение значения epsilon по мере прохождения эпизодов
    epsilon = max(epsilon_final, epsilon_initial - (epsilon_initial - epsilon_final) * episode / epsilon_decay_steps)

    # Вывод результатов эпизода
    print("Episode: {}, Steps: {}, Reward: {:.2f}, Loss: {:.4f}".format(
        episode, episode_steps, episode_reward, episode_loss))

# Сохранение модели для последующего использования
model.save("dqn_model.h5")
