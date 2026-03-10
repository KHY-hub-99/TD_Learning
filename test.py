import random
import numpy as np

# 환경 설정
class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.wall = self.make_wall()

    def make_wall(self):
        wall = [[[1 for _ in range(4)] for _ in range(4)] for _ in range(4)]
        
        target_count = random.choice([2, 3])
        placed_count = 0

        while placed_count < target_count:
            x = random.randint(0, 3)
            y = random.randint(0, 3)
            wall_idx = random.randint(0, 3)

            if (x == 0 and y == 0) or (x == 3 and y == 3):
                continue

            if wall_idx == 0 and y == 0:     # 가장 왼쪽 열의 왼쪽 벽
                continue
            if wall_idx == 1 and x == 0:     # 가장 위쪽 행의 위쪽 벽
                continue
            if wall_idx == 2 and y == 3:     # 가장 오른쪽 열의 오른쪽 벽
                continue
            if wall_idx == 3 and x == 3:     # 가장 아래쪽 행의 아래쪽 벽
                continue

            if wall[x][y][wall_idx] == 0:
                continue
            
            # 바깥 쪽 변 막기
            if (x == 1 and y == 0 and wall_idx == 1) or (x == 0 and y == 1 and wall_idx == 0):
                continue

            if (x == 3 and y == 2 and wall_idx == 2) or (x == 2 and y == 3 and wall_idx == 3):
                continue

            wall[x][y][wall_idx] = 0

            # 양방향
            if wall_idx == 0:
                wall[x][y-1][2] = 0 
            elif wall_idx == 1:
                wall[x-1][y][3] = 0 
            elif wall_idx == 2:
                wall[x][y+1][0] = 0 
            elif wall_idx == 3:
                wall[x+1][y][1] = 0

            placed_count += 1

        return wall

    def is_wall(self, a):
        if self.wall[self.x][self.y][a] == 0:
            return True
        return False
        
    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if not self.is_wall(a):
            if a==0:
                self.move_left()
            elif a==1:
                self.move_up()
            elif a==2:
                self.move_right()
            elif a==3:
                self.move_down()

        reward = -1
        done = self.is_done()

        return reward, done

    def move_right(self):
        self.y += 1  
        if self.y > 3:
            self.y = 3
    
    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0
    
    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else :
            return False

    def get_state(self):
        return (self.x, self.y)
    
    def reset(self):
        self.x = 0
        self.y = 0
        self.wall = self.make_wall()
        return (self.x, self.y)

class Agent():
    def __init__(self):
        pass        

    def select_action(self):
        action = random.choice([0, 1, 2, 3])
        return action
    
# 메인 실행부
def main():
    # TD Learning 초기화
    env = GridWorld()
    agent = Agent()
    data = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]
    gamma = 1.0
    alpha = 0.01

    for k in range(10000):
        done = False
        while not done:
            x, y = env.get_state()
            action = agent.select_action()
            reward, done = env.step(action)
            
            x_prime, y_prime = env.get_state()
            # TD 업데이트 로직
            data[x][y] = data[x][y] + alpha * (reward + gamma * data[x_prime][y_prime] - data[x][y])
        
        # [추가된 로직] 1000번째 에피소드마다 결과 출력
        if (k + 1) % 1000 == 0:
            print(f"\n========== [에피소드 {k + 1} 완료] ==========")
            
            # 1. 벽 위치 출력 (가독성을 위해 막힌 경로만 추출)
            blocked_paths = []
            for i in range(4):
                for j in range(4):
                    if j < 3 and env.wall[i][j][2] == 0: # 오른쪽 경로 막힘
                        blocked_paths.append(f"({i}, {j}) ↔ ({i}, {j+1})")
                    if i < 3 and env.wall[i][j][3] == 0: # 아래쪽 경로 막힘
                        blocked_paths.append(f"({i}, {j}) ↔ ({i+1}, {j})")
            
            print("[현재 막힌 경로]")
            if not blocked_paths:
                print("막힌 곳 없음")
            else:
                for path in blocked_paths:
                    print("- " + path)
            
            # 2. 상태 가치(Value) 소수점 둘째 자리까지 출력
            print("\n[현재 상태 가치 (Value)]")
            current_value = [[round(val, 2) for val in row] for row in data]
            for row in current_value:
                print(row)
        
        # 에피소드 종료 후 환경 리셋
        env.reset()
    
    # 최종 결과 출력
    print("\n========== [최종 학습 결과] ==========")
    final_data = [[round(val, 2) for val in row] for row in data]
    for row in final_data:
        print(row)

if __name__ == '__main__':
    main()