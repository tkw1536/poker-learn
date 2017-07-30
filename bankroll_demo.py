from pklearn import Table
import pickle
from pklearn.templates import simulate

if __name__ == '__main__':

    try:
        import matplotlib.pyplot as plt
    except:
        print 'Must install matplotlib to run this demo.\n'

    t = Table(smallBlind=1, bigBlind=2, maxBuyIn=200)

    players = []
    for i in range(6):
        players.append(pickle.load(open("player{}.player".format(i))))

    for p in players: t.addPlayer(p)
    for p in players: p.setBankroll(2 * (10 ** 5))


    # simulate 20,000 hands and save bankroll history
    bankrolls = simulate(t, nHands=20000, nTrain=0, nBuyIn=10)

    # plot bankroll history of each player
    for i in range(6):
        bankroll = bankrolls[i]
        plt.plot(range(len(bankroll)), bankroll, label=players[i].getName())
    plt.title('Player bankroll vs Hands played')
    plt.xlabel('Hands played')
    plt.ylabel('Player bankroll/wealth')
    plt.legend(loc='upper left')
    plt.show()
