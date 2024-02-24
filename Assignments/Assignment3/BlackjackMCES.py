import numpy as np
import matplotlib.pyplot as plt

# Constants
RANKS = np.arange(1, 11)  # Card ranks 1-10, where 1 is Ace and 10 includes face cards
ACTIONS = [0, 1]  # 0: stick, 1: hit
STATE_SPACE_SIZE = (10, 10, 2)  # Player's sum (12-21), Dealer's showing card (Ace-10), Usable ace (No-Yes)
ACTION_SPACE_SIZE = len(ACTIONS)

# Mapping from card rank to value
card_values = {1: [1, 11]}  # Ace can be 1 or 11
card_values.update({rank: [rank] for rank in range(2, 11)})  # 2-10

# Initialization
Q = np.zeros((*STATE_SPACE_SIZE, ACTION_SPACE_SIZE))  # Action-value function
Returns = [[[] for _ in range(ACTION_SPACE_SIZE)] for _ in range(np.prod(STATE_SPACE_SIZE))]  # Returns for state-action pairs
Policy = np.zeros(STATE_SPACE_SIZE, dtype=int)  # Initial policy (0: stick, 1: hit for all states)

# Function to deal a card
def deal_card():
    """
    Simulates dealing a card. Each rank has an equal probability of being dealt.
    
    Returns:
        int: The rank of the dealt card.
    """
    return np.random.choice(RANKS)

# Function to calculate total and check for usable ace
def hand_value(hand):
    """
    Calculates the total value of a hand and checks for the presence of a usable ace.
    
    Args:
        hand (list of int): The cards in the player's hand.
        
    Returns:
        tuple: Total value of the hand and a boolean indicating the presence of a usable ace.
    """
    total = sum(card_values[card][0] for card in hand)
    usable_ace = any(card == 1 for card in hand) and total + 10 <= 21
    if usable_ace:
        return total + 10, usable_ace
    return total, usable_ace

# Function to play out an episode
def play_episode():
    """
    Plays out a single episode of blackjack following the current policy and dealer's fixed strategy.
    
    Returns:
        list of tuples: The episode as a sequence of (state, action) pairs.
        int: The reward from the episode (1 for win, 0 for draw, -1 for loss).
    """
    episode = []
    player_hand = [deal_card(), deal_card()]
    dealer_hand = [deal_card(), deal_card()]
    
    player_total, usable_ace = hand_value(player_hand)
    dealer_showing = dealer_hand[0]
    
    # Ensure player's total is at least 12
    while player_total < 12:
        player_hand.append(deal_card())
        player_total, usable_ace = hand_value(player_hand)
    
    state = (player_total - 12, dealer_showing - 1, int(usable_ace))
    action = np.random.choice(ACTIONS)  # First action is chosen randomly
    episode.append((state, action))
    
    # Player's turn
    while action == 1 and player_total < 21:  # Hit
        player_hand.append(deal_card())
        player_total, usable_ace = hand_value(player_hand)
        state = (player_total - 12, dealer_showing - 1, int(usable_ace))
        if player_total < 21:
            action = Policy[state]  # Follow current policy
            episode.append((state, action))
        else:
            break  # Player goes bust
    
    # Dealer's turn
    if player_total <= 21:
        dealer_total, _ = hand_value(dealer_hand)
        while dealer_total < 17:  # Dealer hits on 16 or less
            dealer_hand.append(deal_card())
            dealer_total, _ = hand_value(dealer_hand)
    
    # Determine reward
    reward = 0
    if player_total > 21:  # Player bust
        reward = -1
    elif dealer_total > 21 or player_total > dealer_total:  # Dealer bust or player has higher total
        reward = 1
    elif player_total < dealer_total:
        reward = -1  # Dealer wins
    
    return episode, reward

# Function to update Q and policy
def update_policy(episode, reward):
    """
    Updates the policy based on the episode played and the reward received.
    
    Args:
        episode (list of tuples): The episode as a sequence of (state, action) pairs.
        reward (int): The reward from the episode.
    """
    # Policy evaluation
    for state, action in episode:
        state_index = np.ravel_multi_index(state, STATE_SPACE_SIZE)
        Returns[state_index][action].append(reward)
        Q[state][action] = np.mean(Returns[state_index][action])
    
    # Policy improvement
    for state, _ in episode:
        Policy[state] = np.argmax(Q[state])

# Function to track Q(s, stick) for specific state over episodes
def track_Q_over_episodes(n_episodes, track_state=(18-12, 8-1, 0)):
    """
    Tracks the Q-value for a specific state (player total = 18, dealer showing = 8, no usable ace) over multiple episodes.
    
    The state is represented as (18-12, 8-1, 0) for zero-based indexing in the simulation:
    - The player's total is adjusted from 18 to a zero-based index by subtracting 12, resulting in 6. This is because the player's decision-making process starts from a total of 12 in the simulation.
    - The dealer's showing card is adjusted from 8 to a zero-based index by subtracting 1, resulting in 7. This aligns with the array indexing in Python, which starts at 0.
    - The last element, 0, indicates the absence of a usable ace in the player's hand.
    
    This approach ensures the state is correctly identified within the simulation's indexing system, allowing for accurate tracking and updating of the Q-value associated with sticking in this specific state across different episodes.
    
    Args:
        n_episodes (int): The number of episodes to simulate.
        track_state (tuple): The state to track, adjusted for zero-based indexing.
        
    Returns:
        np.ndarray: The episodes at which Q-value is recorded.
        list: The Q-values recorded at those episodes.
    """
    Q_values = [] # Q(s, stick) for the specific state over episodes
    episode_counts = np.arange(100, n_episodes + 1, 100) # Track Q(s, stick) over 100 episodes up to n_episodes
    
    # Loop over episodes and update Q and policy accordingly, and track Q(s, stick) for the specific state
    for episode in range(1, n_episodes + 1):
        ep, reward = play_episode()
        update_policy(ep, reward)
        if episode in episode_counts:
            Q_values.append(Q[track_state][0])  # Q(s, stick) when player state is (18, 8, no usable ace)
    
    return episode_counts, Q_values

# Track Q(s, stick) over 10,000 episodes for each 100 episodes
episode_counts, Q_values = track_Q_over_episodes(10000)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(episode_counts, Q_values, marker='.', linestyle='-', color='r')
# make the plot more readable by making the plot larger

plt.title('Q(s, stick) over Episodes for s=(18, 8, no usable ace)')
plt.xlabel('Episodes')
plt.ylabel('Q(s, stick)')
plt.grid(True)
plt.show()




