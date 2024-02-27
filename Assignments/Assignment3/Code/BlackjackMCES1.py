import numpy as np
import matplotlib.pyplot as plt
import time # to measure time taken for 100,000 episodes

# Constants
ACTION_HIT = 1
ACTION_STICK = 0
ACTIONS = [ACTION_STICK, ACTION_HIT]
SPECIFIC_STATE = (18, 8, False)  # Specific state to check the Q value for
SPECIFIC_ACTION = ACTION_STICK  # Specific action to check the Q value for
policy = {}
for player_sum in range(12, 22):
    for dealer_card in range(1, 11):
        for usable_ace in (False, True):
            # Policy from the book
            policy[(player_sum, dealer_card, usable_ace)] = ACTION_STICK if player_sum >= 20 else ACTION_HIT

            # Random policy
            # policy[(player_sum, dealer_card, usable_ace)] = np.random.choice(ACTIONS)
Q = {}
Returns = {}
for state in policy.keys():
    for action in ACTIONS:
        Q[(state, action)] = 0
        Returns[(state, action)] = []

# Function Definitions
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

def play_episode():
    player_cards = [deal_card(), deal_card()]
    dealer_cards = [deal_card(), deal_card()]
    state = (total_hand(player_cards), dealer_cards[0], has_usable_ace(player_cards))
    
    # Ensure player's total is at least 12 to start
    while total_hand(player_cards) < 12:
        player_cards.append(deal_card())
        state = (total_hand(player_cards), dealer_cards[0], has_usable_ace(player_cards))
    
    episode = []
    
    # Player's turn
    while True:
        if state not in policy:
            break
        action = policy[state]
        episode.append((state, action))
        if action == ACTION_STICK:
            break
        player_cards.append(deal_card())
        player_total = total_hand(player_cards)
        if player_total > 21:
            return episode, -1  # Player busts
        state = (player_total, dealer_cards[0], has_usable_ace(player_cards))
    
    # Dealer's turn
    while total_hand(dealer_cards) < 17:
        dealer_cards.append(deal_card())
    dealer_total = total_hand(dealer_cards)
    if dealer_total > 21 or dealer_total < state[0]:
        return episode, 1  # Dealer busts or player has higher total
    elif dealer_total == state[0]:
        return episode, 0  # Draw
    else:
        return episode, -1  # Dealer wins

def monte_carlo_ES(episodes):
    Q_values = []
    for episode_num in range(1, episodes + 1):
        episode, reward = play_episode()
        for state, action in episode:
            Returns[(state, action)].append(reward)
            Q[(state, action)] = np.mean(Returns[(state, action)])
            # Greedily update the policy for the state
            policy[state] = np.argmax([Q[(state, a)] for a in ACTIONS])
        if episode_num % 100 == 0:
            Q_values.append(Q[(SPECIFIC_STATE, SPECIFIC_ACTION)])
    return Q_values

def play_game_episodes(episodes):
    wins = 0
    for _ in range(episodes):
        _, result = play_episode()
        if result == 1:  # Win condition
            wins += 1
    win_percentage = (wins / episodes) * 100
    return wins, win_percentage

# Run Monte Carlo ES for 10,000 episodes, and check how much time it takes
start_time = time.time()
Q_values = monte_carlo_ES(10000)
end_time = time.time()
print("Training time taken for 10,000 episodes: ", end_time - start_time)

# Plotting
plt.figure(figsize=(14, 7))
plt.title(f"Values of Q(s, a) for state {SPECIFIC_STATE} and action {SPECIFIC_ACTION}")
plt.xlabel("Episodes")
plt.ylabel("Q(s, a)")
plt.plot(range(100, 10001, 100), Q_values, marker='.', linestyle='-', color='r')
plt.grid(True, which="both", ls="-")
plt.tight_layout()
# plt.savefig("Assignments/Assignment3/Report/Q_values.png")
plt.show()

print("Going to play 100,000 episodes to check the winning percentage")

L = 100000
wins, win_percentage = play_game_episodes(L)
# plot the winning times for 100,000 episodes (binary plot)
plt.figure(figsize=(14, 7))
plt.title("Winning Times for 100,000 episodes")
plt.xlabel("Episodes")
plt.ylabel("Winning Times")
plt.plot(range(1, L + 1), [1 if i <= wins else 0 for i in range(1, L + 1)], marker='.', linestyle='-', color='orange')  
plt.grid(True, which="both", ls="-")
plt.text(0.5, 0.9, f"Winning Percentage: {win_percentage:.2f}%", ha='center', va='center', transform=plt.gca().transAxes)
plt.tight_layout()
# plt.savefig("Assignments/Assignment3/Report/MCES1Winning_Times.png")
plt.show()


