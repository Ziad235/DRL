import numpy as np
import matplotlib.pyplot as plt
import time # to measure time taken for 100,000 episodes

# Constants
ACTION_HIT = 1
ACTION_STICK = 0
ACTIONS = [ACTION_STICK, ACTION_HIT]
EPSILON = 0.1
L = 2000  # Number of episodes between tests
ALPHAS = [0.01, 0.1]
TEST_EPISODES = 10000

# Function Definitions (deal_card, has_usable_ace, total_hand) remain unchanged
def deal_card():
    """Returns a card from the deck. Cards are 1-10, with face cards being 10."""
    return min(np.random.randint(1, 14), 10)

def has_usable_ace(cards):
    """Returns True if the hand has a usable ace."""
    return 1 in cards and sum(cards) + 10 <= 21

def total_hand(cards):
    """Returns the total of a hand, counting aces as 11 if they don't bust the hand."""
    total = sum(cards)
    if has_usable_ace(cards):
        return total + 10
    return total

def choose_action(state, Q, epsilon=EPSILON):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        return np.argmax([Q.get((state, a), 0) for a in ACTIONS])

def update_Q(Q, state, action, reward, next_state, alpha, done):
    next_max = 0 if done else max([Q.get((next_state, a), 0) for a in ACTIONS])
    Q[(state, action)] = Q.get((state, action), 0) + alpha * (reward + next_max - Q.get((state, action), 0))

def play_episode(Q, alpha, epsilon=EPSILON):
    player_cards = [deal_card(), deal_card()]
    dealer_cards = [deal_card(), deal_card()]
    state = (total_hand(player_cards), dealer_cards[0], has_usable_ace(player_cards))
    while total_hand(player_cards) < 12:
        player_cards.append(deal_card())
        state = (total_hand(player_cards), dealer_cards[0], has_usable_ace(player_cards))
    done = False
    while not done:
        action = choose_action(state, Q, epsilon)
        if action == ACTION_STICK:
            break
        player_cards.append(deal_card())
        player_total = total_hand(player_cards)
        if player_total > 21:
            update_Q(Q, state, action, -1, None, alpha, done=True)
            return -1  # Player busts
        next_state = (player_total, dealer_cards[0], has_usable_ace(player_cards))
        update_Q(Q, state, action, 0, next_state, alpha, done=False)
        state = next_state
    # Dealer's turn
    while total_hand(dealer_cards) < 17:
        dealer_cards.append(deal_card())
    dealer_total = total_hand(dealer_cards)
    if dealer_total > 21 or dealer_total < state[0]:
        reward = 1
    elif dealer_total == state[0]:
        reward = 0
    else:
        reward = -1
    update_Q(Q, state, ACTION_STICK, reward, None, alpha, done=True)
    return reward

def test_policy(Q):
    wins = 0
    for _ in range(TEST_EPISODES):
        result = play_episode(Q, alpha=0, epsilon=0)  # Test with greedy policy
        if result == 1:
            wins += 1
    return wins / TEST_EPISODES

def train_and_test(alpha, episodes=100000):
    Q = {}
    test_results = []
    specific_Q_values = []  # To track Q(s, STICK) for the specific state over episodes
    specific_state = (18, 8, False)  # The specific state to track
    specific_action = ACTION_STICK  # The specific action to track (STICK)
    
    for i in range(1, episodes + 1):
        play_episode(Q, alpha)
        # Track Q-value for the specific state-action pair
        specific_Q_values.append(Q.get((specific_state, specific_action), 0))
        
        if i % L == 0:
            win_rate = test_policy(Q)
            test_results.append(win_rate)
            print(f"Alpha: {alpha}, Episode: {i}, Win rate: {win_rate*100:.2f}%, Q(18,8,False,STICK): {specific_Q_values[-1]}")

    # set the color for the graph
    graph_color = 'g' if alpha == 0.01 else 'r'
    
    # Plotting the Q-value for the specific state-action pair over episodes
    plt.figure(figsize=(13, 7))
    plt.plot(range(1, episodes + 1), specific_Q_values, label=f'Q(s=(18, 8, False), a=STICK), Alpha = {alpha}', color=graph_color)
    plt.title('Q-value of Specific State-Action Pair Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Q(s=(18, 8, False), a=STICK)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Assignments/Assignment3/Report/Q_Blackjack_Alpha_{alpha}.png')
    plt.show()
    
    # Plotting win rates after test episodes as before
    plt.figure(figsize=(13, 7))
    plt.plot(range(L, episodes + 1, L), test_results, label=f'Test Win Rate, Alpha = {alpha}', color=graph_color)
    plt.title('Test Win Rate Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'Assignments/Assignment3/Report/Win_Rate_Blackjack_Alpha_{alpha}.png')
    plt.show()


# Train and test for different alphas
# test time taken for 100000 episodes
for alpha in ALPHAS:
    start_time = time.time()
    train_and_test(alpha, episodes=100000)
    end_time = time.time()
    print(f"Training time taken for 100,000 episodes with alpha = {alpha}: {end_time - start_time:.2f} seconds")



