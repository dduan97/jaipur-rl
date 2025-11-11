import random

_DIAMOND = "DIAMOND"
_GOLD = "GOLD"
_SILVER = "SILVER"
_SILK = "SILK"
_SPICE = "SPICE"
_LEATHER = "LEATHER"
_CAMEL = "CAMEL"


class Player:
    def __init__(self, p_id):
        self.id = p_id
        # self.soe = 0
        # self.reset()

    def reset(self):
        self.cards = {
            _DIAMOND: 0,
            _GOLD: 0,
            _SILVER: 0,
            _SILK: 0,
            _SPICE: 0,
            _LEATHER: 0,
            _CAMEL: 0,
        }
        self.bonus = 0
        self.opp_points = 0
        self.opp_herd_size = 0

    def add_good(self, good):
        self.cards[good] += 1

    def hand_size(self):
        return sum(v for k, v in self.cards.items() if k != _CAMEL)

    def herd_size(self):
        return self.cards[_CAMEL]


class GoodsDeck:
    def __init__(self, goods_counts: dict | None = None):
        # deck contains total of 55 cards
        # leather, spices, silk, silver, gold, diamonds, camel
        self.cards = []

        # 11 camel cards
        # 6 diamonds
        # 6 gold
        # 6 silver
        # 8 silk
        # 8 spice
        # 10 leather
        goods_counts = goods_counts or {
            _CAMEL: 11,
            _DIAMOND: 6,
            _GOLD: 6,
            _SILVER: 6,
            _SILK: 8,
            _SPICE: 8,
            _LEATHER: 10,
        }

        for goods_type, count in goods_counts.items():
            self.cards.extend([goods_type for _ in range(count)])

        self.cards.sort()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        # taking one card from the deck of cards
        if len(self.cards) > 0:
            return self.cards.pop(0)
        return None

    def deal_camel(self):
        for i in range(len(self.cards)):
            if self.cards[i] == _CAMEL:
                return self.cards.pop(i)


class TokenDeck:
    def __init__(self):
        # 60 tokens

        # 38 goods
        # 18 bonus
        # 1 camel
        # 3 soe
        self.goods_type_tokens = {
            _LEATHER: [4, 3, 2, 1, 1, 1, 1, 1, 1],
            _SPICE: [5, 3, 3, 2, 2, 1, 1],
            _SILK: [5, 3, 3, 2, 2, 1, 1],
            _SILVER: [5, 5, 5, 5, 5],
            _GOLD: [6, 6, 5, 5, 5],
            _DIAMOND: [7, 7, 5, 5, 5],
        }

        # 18 total bonus tokens
        self.bonus_tokens = {
            3: [1, 1, 2, 2, 2, 3, 3],
            4: [4, 4, 5, 5, 6, 6],
            5: [8, 8, 9, 10, 10],
        }

        # bonus tokens should be shuffled
        for tokens in self.bonus_tokens.values():
            random.shuffle(tokens)

        # self.soe = ["S1", "S2", "S3"] #3
        self.camel_token = 5
        # leather, spices, silk, silver, gold, diamonds, bonus, seals of excellence, camel

        self.empty = []


class Jaipur:
    def __init__(self):
        self.p1 = Player(0)
        self.p2 = Player(1)

        self.create_trade_dict()

        # self.round()

    def create_trade_dict(self):
        # creating trade actions dictionary of 25,456 actions

        in_hand = [_DIAMOND, _GOLD, _SILVER, _SPICE, _SILK, _LEATHER, _CAMEL]
        marketplace_cards = [_DIAMOND, _GOLD, _SILVER, _SPICE, _SILK, _LEATHER]

        self.Q_trade = {}
        count = 13

        # trading 2 cards
        for i in range(len(in_hand) - 1, -1, -1):
            for j in range(i, -1, -1):

                for a in range(len(marketplace_cards) - 1, -1, -1):
                    for b in range(a, -1, -1):

                        if (
                            marketplace_cards[a] != in_hand[i]
                            and marketplace_cards[a] != in_hand[j]
                            and marketplace_cards[b] != in_hand[i]
                            and marketplace_cards[b] != in_hand[j]
                        ):
                            self.Q_trade[count] = (
                                (in_hand[i], in_hand[j], "na", "na", "na"),
                                (
                                    marketplace_cards[a],
                                    marketplace_cards[b],
                                    "na",
                                    "na",
                                    "na",
                                ),
                            )  # = 0
                            count = count + 1

        # trading 3 cards
        for i in range(len(in_hand) - 1, -1, -1):
            for j in range(i, -1, -1):
                for k in range(j, -1, -1):

                    for a in range(len(marketplace_cards) - 1, -1, -1):
                        for b in range(a, -1, -1):
                            for c in range(b, -1, -1):

                                if (
                                    in_hand[i] != marketplace_cards[a]
                                    and in_hand[j] != marketplace_cards[a]
                                    and in_hand[k] != marketplace_cards[a]
                                    and in_hand[i] != marketplace_cards[b]
                                    and in_hand[j] != marketplace_cards[b]
                                    and in_hand[k] != marketplace_cards[b]
                                    and in_hand[i] != marketplace_cards[c]
                                    and in_hand[j] != marketplace_cards[c]
                                    and in_hand[k] != marketplace_cards[c]
                                ):
                                    self.Q_trade[count] = (
                                        (
                                            in_hand[i],
                                            in_hand[j],
                                            in_hand[k],
                                            "na",
                                            "na",
                                        ),
                                        (
                                            marketplace_cards[a],
                                            marketplace_cards[b],
                                            marketplace_cards[c],
                                            "na",
                                            "na",
                                        ),
                                    )  # = 0
                                    count = count + 1

        # trading 4 cards
        for i in range(len(in_hand) - 1, -1, -1):
            for j in range(i, -1, -1):
                for k in range(j, -1, -1):
                    for l in range(k, -1, -1):

                        for a in range(len(marketplace_cards) - 1, -1, -1):
                            for b in range(a, -1, -1):
                                for c in range(b, -1, -1):
                                    for d in range(c, -1, -1):

                                        if (
                                            in_hand[i] != marketplace_cards[a]
                                            and in_hand[j] != marketplace_cards[a]
                                            and in_hand[k] != marketplace_cards[a]
                                            and in_hand[l] != marketplace_cards[a]
                                            and in_hand[i] != marketplace_cards[b]
                                            and in_hand[j] != marketplace_cards[b]
                                            and in_hand[k] != marketplace_cards[b]
                                            and in_hand[l] != marketplace_cards[b]
                                            and in_hand[i] != marketplace_cards[c]
                                            and in_hand[j] != marketplace_cards[c]
                                            and in_hand[k] != marketplace_cards[c]
                                            and in_hand[l] != marketplace_cards[c]
                                            and in_hand[i] != marketplace_cards[d]
                                            and in_hand[j] != marketplace_cards[d]
                                            and in_hand[k] != marketplace_cards[d]
                                            and in_hand[l] != marketplace_cards[d]
                                        ):
                                            self.Q_trade[count] = (
                                                (
                                                    in_hand[i],
                                                    in_hand[j],
                                                    in_hand[k],
                                                    in_hand[l],
                                                    "na",
                                                ),
                                                (
                                                    marketplace_cards[a],
                                                    marketplace_cards[b],
                                                    marketplace_cards[c],
                                                    marketplace_cards[d],
                                                    "na",
                                                ),
                                            )  # = 0
                                            count = count + 1

        # trading 5 cards
        for i in range(len(in_hand) - 1, -1, -1):
            for j in range(i, -1, -1):
                for k in range(j, -1, -1):
                    for l in range(k, -1, -1):
                        for m in range(l, -1, -1):

                            for a in range(len(marketplace_cards) - 1, -1, -1):
                                for b in range(a, -1, -1):
                                    for c in range(b, -1, -1):
                                        for d in range(c, -1, -1):
                                            for e in range(d, -1, -1):

                                                if (
                                                    in_hand[i] != marketplace_cards[a]
                                                    and in_hand[j]
                                                    != marketplace_cards[a]
                                                    and in_hand[k]
                                                    != marketplace_cards[a]
                                                    and in_hand[l]
                                                    != marketplace_cards[a]
                                                    and in_hand[m]
                                                    != marketplace_cards[a]
                                                    and in_hand[i]
                                                    != marketplace_cards[b]
                                                    and in_hand[j]
                                                    != marketplace_cards[b]
                                                    and in_hand[k]
                                                    != marketplace_cards[b]
                                                    and in_hand[l]
                                                    != marketplace_cards[b]
                                                    and in_hand[m]
                                                    != marketplace_cards[b]
                                                    and in_hand[i]
                                                    != marketplace_cards[c]
                                                    and in_hand[j]
                                                    != marketplace_cards[c]
                                                    and in_hand[k]
                                                    != marketplace_cards[c]
                                                    and in_hand[l]
                                                    != marketplace_cards[c]
                                                    and in_hand[m]
                                                    != marketplace_cards[c]
                                                    and in_hand[i]
                                                    != marketplace_cards[d]
                                                    and in_hand[j]
                                                    != marketplace_cards[d]
                                                    and in_hand[k]
                                                    != marketplace_cards[d]
                                                    and in_hand[l]
                                                    != marketplace_cards[d]
                                                    and in_hand[m]
                                                    != marketplace_cards[d]
                                                    and in_hand[i]
                                                    != marketplace_cards[e]
                                                    and in_hand[j]
                                                    != marketplace_cards[e]
                                                    and in_hand[k]
                                                    != marketplace_cards[e]
                                                    and in_hand[l]
                                                    != marketplace_cards[e]
                                                    and in_hand[m]
                                                    != marketplace_cards[e]
                                                ):
                                                    self.Q_trade[count] = (
                                                        (
                                                            in_hand[i],
                                                            in_hand[j],
                                                            in_hand[k],
                                                            in_hand[l],
                                                            in_hand[m],
                                                        ),
                                                        (
                                                            marketplace_cards[a],
                                                            marketplace_cards[b],
                                                            marketplace_cards[c],
                                                            marketplace_cards[d],
                                                            marketplace_cards[e],
                                                        ),
                                                    )  # = 0
                                                    count = count + 1

    def round(self):
        self.goods = GoodsDeck()
        self.tokens = TokenDeck()

        self.p1.reset()
        self.p2.reset()

        self.marketplace = []
        self.discard = []

        # adding 3 camel cards to the marketplace
        for i in range(3):
            self.marketplace.append(self.goods.deal_camel())

        # shuffle cards
        self.goods.shuffle()

        # adding 2 cards from shuffled cards to the marketplace
        for i in range(2):
            self.marketplace.append(self.goods.deal())

        print("Marketplace:", self.marketplace)

        # giving out 5 cards to each player
        for p in self.p1, self.p2:
            for _ in range(5):
                # pop card from deck
                c = self.goods.deal()
                p.add_good(c)

        print("Player 1 cards: ", self.p1.cards)
        print("Player 2 cards: ", self.p2.cards)

        # setting opponent's herd size
        self.p1.opp_herd_size = self.p2.herd_size()
        self.p2.opp_herd_size = self.p1.herd_size()

    def final(self):
        # checking for which player had the most camels
        if self.p1.herd_size() != self.p2.herd_size():
            camel_bonus_player = self.p1 if self.p1.herd_size() > self.p2.herd_size() else self.p2
            camel_bonus_player.bonus += self.tokens.camel_token

        print(
            "Player 1 has ",
            self.p1.bonus,
            " points.\nPlayer 2 has ",
            self.p2.bonus,
            " points.",
        )

    def take_1_good(self, chosen, player):
        # checking that the player's hand is not 7 or greater
        if player.hand > 6:
            # displaying error message
            # print("ERROR! Unable to take card as hand size must never be more than 7 cards.")
            return

        # making sure that the chosen card is not a camel card
        if chosen != _CAMEL:
            # setting found to False
            found = False
            # looping to find chosen card to take
            for i in range(len(self.marketplace)):
                # if card is found
                if self.marketplace[i] == chosen:
                    # removing card from marketplace
                    card = self.marketplace.pop(i)
                    # adding card to player's cards
                    if card in player.cards:
                        player.cards[card] = player.cards[card] + 1
                        # increasing player hand amount
                        player.hand = player.hand + 1

                    # setting found to True
                    found = True

                    # getting the next card from the deck
                    new_card = self.cards.deal()

                    # if there was a card in the deck
                    if new_card != None:
                        # adding one card from deck to marketplace
                        self.marketplace.append(new_card)

                    # exiting out of the loop
                    break
            # displaying error message if the chosen card was not found in the marketplace
            if found == False:
                # print("\nERROR! The chosen card: ",chosen, " was not available in the marketplace.")
                return
        else:
            # print("\nERROR! The chosen card must be a goods card")
            return

    def take_goods(self, chosen, player, replacable):
        # checking that the player is replacing all cards taken
        if len(chosen) != len(replacable):
            # print("You must replace all the cards taken!")
            return

        # to keep track of how many camels the player is going to use to replace the cards
        camels = 0

        p_cards = player.cards.copy()

        # looping to check that all the cards are in the player's hand or herd
        for k in range(len(replacable)):
            # if the card is not found in the player's cards (hand/herd)
            if p_cards[replacable[k]] == 0:
                # if not (replacable[k] in player.hand or replacable[k] in player.herd):
                # displaying error message
                # print("ERROR! The card: ", replacable[k], " is not present in your hand/herd.")
                # exiting the function
                return
            # checking for Camel cards
            elif replacable[k] == _CAMEL:
                # increasing the count by 1 if a camel is found
                camels = camels + 1
            # decreasing the value of the card in temporary dictionary
            # to make sure that the player has enough cards
            p_cards[replacable[k]] = p_cards[replacable[k]] - 1

        # checking the new hand size
        if ((player.hand - (len(chosen) - camels)) + len(replacable)) > 7:
            # displaying error message if hand size will be greater than 7
            # print("ERROR! Unable to trade these cards as hand size must never be more than 7 cards.")
            # exiting the function
            return
        # print("Player hand: ", player.hand)

        for i in chosen:
            if i in replacable:
                # print("ERROR! Unable to trade cards of same card type.")
                return

        if _CAMEL in chosen:
            # print("ERROR! The chosen cards must be goods cards not camel cards! ")
            return

        temp = self.marketplace.copy()
        found = False
        for i in chosen:
            for j in range(len(temp) - 1, -1, -1):
                if i == temp[j]:
                    temp.pop(j)
                    found = True
                    break
            if found == False:
                # print("ERROR! The chosen cards were not available in the marketplace.")
                return

        player.hand = (player.hand - (len(chosen) - camels)) + len(replacable)
        # print("Player hand: ", player.hand)

        # looping backwards to find the chosen cards to take
        for i in range(len(chosen) - 1, -1, -1):
            # looping backwards in the marketplace to search for the chosen cards
            for j in range(len(self.marketplace) - 1, -1, -1):
                # if card is found
                if self.marketplace[j] == chosen[i]:
                    # removing card from marketplace
                    card = self.marketplace.pop(j)
                    # adding card to player's hand
                    player.cards[card] = player.cards[card] + 1
                    # player.hand.append(card)

                    # removing card from chosen array
                    chosen.pop(i)

                    # adding the replacement card to the marketplace
                    self.marketplace.append(replacable[i])
                    # removing replacement card from player's cards
                    player.cards[replacable[i]] = player.cards[replacable[i]] - 1

                    # exiting out of the loop to move onto next chosen card
                    break

    def take_camels(self, player):
        # to keep track of how many camels were taken
        count = 0
        # looping backwards in the marketplace to search for camels
        for i in range(len(self.marketplace) - 1, -1, -1):
            # if a camel card is found
            if self.marketplace[i] == _CAMEL:
                # removing camel card from marketplace
                card = self.marketplace.pop(i)
                # adding camel card to player's herd
                player.cards[card] = player.cards[card] + 1
                player.herd = player.herd + 1
                # player.herd.append(card)
                # increasing count
                count = count + 1

        # looping to add cards to marketplace according to how many camels were taken
        # after camels were taken
        for i in range(count):
            if len(self.marketplace) >= 5:
                # print("ERROR! Exceeding marketplace size.")
                return
            # getting the next card from the deck
            new_card = self.cards.deal()

            # if there was a card in the deck
            if new_card != None:
                # adding the card to the marketplace
                self.marketplace.append(new_card)

    def sell_goods(self, chosen, player):
        # checking that the player is selling at least 2 cards if of type silver, gold or diamond
        if len(chosen) < 2 and (
            chosen[0] == _SILVER or chosen[0] == _GOLD or chosen[0] == _DIAMOND
        ):
            # displaying error message if the player is trying to sell less than 2 cards
            # print("ERROR! Must sell at least 2 cards when selling type ", chosen[0], ".")
            # exiting function
            return

        # looping to check that all cards are of the same type
        for i in range(len(chosen) - 1):
            if chosen[0] != chosen[i + 1]:
                # displaying error message if a different type is found
                # print("ERROR! Cards must all be of the same type.")
                # exiting function
                return

        # checking that the player has all cards in his hand
        if player.cards[chosen[0]] < len(chosen):
            # displaying error message if player is trying to sell more cards than he owns
            # print("ERROR! You do not have ", len(chosen), " cards of type ", chosen[0], " in your hand.")
            return

        # to keep track of how many tokens were sold
        count = 0

        # looping through the chosen cards to sell
        for i in range(len(chosen) - 1, -1, -1):
            # getting token amount
            amount = self.ret_token_amount(chosen[0])
            # print("token amount: ", amount)
            # if there aren't anymore tokens left
            # sell card without taking token
            if amount == 0:
                # removing card from player's hand
                self.sell_card(chosen[i], player)
                # increasing count by 1
                count = count + 1

            elif amount != 0:
                # removing card from player's hand
                self.sell_card(chosen[i], player)
                # taking a token and getting it's value
                tok = self.pop_token(chosen[0])
                # adding token value to player's bonus
                player.bonus = player.bonus + tok
                # increasing count by 1
                count = count + 1

                # checking if the token array is empty
                self.check_token(chosen[0])

        # player will know opponent's token points
        if player.id == 0:
            # if p1 sold cards
            self.p2.opp_points = player.bonus
        elif player.id == 1:
            # if p2 sold cardsret_token_amount
            self.p1.opp_points = player.bonus

        # player will receive their appropriate bonus token if they sell more than 3 cards
        bonus = self.pop_bonus_token(count)
        # adding bonus token value to player's bonus
        player.bonus = player.bonus + bonus

    def ret_token_amount(self, c_type):
        # returning the amount of tokens left of the card type
        if c_type == _DIAMOND:
            return len(self.tokens.diamond)
        elif c_type == _GOLD:
            return len(self.tokens.gold)
        elif c_type == _SILVER:
            return len(self.tokens.silver)
        elif c_type == _SILK:
            return len(self.tokens.silk)
        elif c_type == _SPICE:
            return len(self.tokens.spice)
        elif c_type == _LEATHER:
            return len(self.tokens.leather)
        else:
            # print("ERROR! Token type not found.")
            return

    def pop_token(self, c_type):
        # returning the token value depending on the card type
        # token will be removed from the array
        if c_type == _DIAMOND:
            return self.tokens.diamond.pop(0)
        elif c_type == _GOLD:
            return self.tokens.gold.pop(0)
        elif c_type == _SILVER:
            return self.tokens.silver.pop(0)
        elif c_type == _SILK:
            return self.tokens.silk.pop(0)
        elif c_type == _SPICE:
            return self.tokens.spice.pop(0)
        elif c_type == _LEATHER:
            return self.tokens.leather.pop(0)
        else:
            # print("ERROR! Token type not found.")
            return

    def pop_bonus_token(self, count):
        # returning the token value depending on how many cards were sold
        # token will be removed from the array
        if count == 3 and len(self.tokens.bonus_3) > 0:
            return self.tokens.bonus_3.pop(0)
        elif count == 4 and len(self.tokens.bonus_4) > 0:
            return self.tokens.bonus_4.pop(0)
        elif count >= 5 and len(self.tokens.bonus_5) > 0:
            return self.tokens.bonus_5.pop(0)
        else:
            return 0

    def sell_card(self, card, player):
        if card in player.cards:
            if player.cards[card] > 0:
                # removing the card from the players' cards
                player.cards[card] = player.cards[card] - 1
                player.hand = player.hand - 1
                # adding the card to the discard pile
                self.discard.append(card)
            else:
                print("ERROR! Card not found in player's hand.")
                return

    def check_token(self, c_type):
        # checking if there are any more tokens left of the card type
        # if an array is found to be empty, it will be appended to the empty tokens array
        if c_type == _DIAMOND:
            if len(self.tokens.diamond) == 0:
                print("Finished tokens of type: ", c_type)
                self.tokens.empty.append(c_type)
        elif c_type == _GOLD:
            if len(self.tokens.gold) == 0:
                print("Finished tokens of type: ", c_type)
                self.tokens.empty.append(c_type)
        elif c_type == _SILVER:
            if len(self.tokens.silver) == 0:
                print("Finished tokens of type: ", c_type)
                self.tokens.empty.append(c_type)
        elif c_type == _SILK:
            if len(self.tokens.silk) == 0:
                print("Finished tokens of type: ", c_type)
                self.tokens.empty.append(c_type)
        elif c_type == _SPICE:
            if len(self.tokens.spice) == 0:
                print("Finished tokens of type: ", c_type)
                self.tokens.empty.append(c_type)
        elif c_type == _LEATHER:
            if len(self.tokens.leather) == 0:
                print("Finished tokens of type: ", c_type)
                self.tokens.empty.append(c_type)
        else:
            print("ERROR! Token type not found.")
            return

    def finished(self):
        # if at least three goods arrays are empty or marketplace has less than 5 cards, the round is finished
        if (len(self.tokens.empty) >= 3) or (len(self.marketplace) < 5):
            # print("finished",self.tokens.empty, len(self.marketplace), len(self.cards.cards))
            return True
        # otherwise the round continues
        else:
            return False

    def sell_cards(self, player, card_type):
        chosen = []
        if player.cards[card_type] > 0:
            # print("Amount: ", player.cards[card_type])
            for i in range(player.cards[card_type]):
                chosen.append(card_type)
        else:
            # print("ERROR! Must sell at least 1 card.")
            return []

        return chosen

    def sell_min2_cards(self, player, card_type):
        chosen = []
        if player.cards[card_type] > 1:
            # print("Amount: ", player.cards[card_type])
            for i in range(player.cards[card_type]):
                chosen.append(card_type)
        else:
            # print("ERROR! Must sell at least 2 cards when selling type ", card_type,".")
            return []

        return chosen

    def options(self, player, choice):
        # take diamond, take gold, take silver, take silk, take spice, take leather, take_goods, take_camels, sell_goods

        if choice == 0:
            print("Taking 1 Diamond")
            #             card = self.choose_card()
            #             self.take_1_good(card, player)
            self.take_1_good(_DIAMOND, player)
        elif choice == 1:
            print("Taking 1 Gold")
            self.take_1_good(_GOLD, player)
        elif choice == 2:
            print("Taking 1 Silver")
            self.take_1_good(_SILVER, player)
        elif choice == 3:
            print("Taking 1 Silk")
            self.take_1_good(_SILK, player)
        elif choice == 4:
            print("Taking 1 Spice")
            self.take_1_good(_SPICE, player)
        elif choice == 5:
            print("Taking 1 Leather")
            self.take_1_good(_LEATHER, player)
        elif choice == 6:
            print("Taking All Camels")
            self.take_camels(player)
        elif choice >= 7 and choice <= 12:
            chosen = []
            if choice == 7:
                # print("Selling Diamond")
                chosen = self.sell_min2_cards(player, _DIAMOND)
            elif choice == 8:
                # print("Selling Gold")
                chosen = self.sell_min2_cards(player, _GOLD)
            elif choice == 9:
                # print("Selling Silver")
                chosen = self.sell_min2_cards(player, _SILVER)
            elif choice == 10:
                # print("Selling Silk")
                chosen = self.sell_cards(player, _SILK)
            elif choice == 11:
                # print("Selling Spice")
                chosen = self.sell_cards(player, _SPICE)
            elif choice == 12:
                # print("Selling Leather")
                chosen = self.sell_cards(player, _LEATHER)

            if chosen != []:
                print("Selling:", chosen)
                self.sell_goods(chosen, player)

        elif choice > 12 and choice <= 25468:
            # trading goods

            replacable = []
            chosen = []
            for i in self.Q_trade[choice][0]:  # replacable
                if i == _CAMEL:
                    replacable.append(_CAMEL)
                elif i == _LEATHER:
                    replacable.append(_LEATHER)
                elif i == _SPICE:
                    replacable.append(_SPICE)
                elif i == _SILK:
                    replacable.append(_SILK)
                elif i == _SILVER:
                    replacable.append(_SILVER)
                elif i == _GOLD:
                    replacable.append(_GOLD)
                elif i == _DIAMOND:
                    replacable.append(_DIAMOND)

            for j in self.Q_trade[choice][1]:  # chosen
                if j == _LEATHER:
                    chosen.append(_LEATHER)
                elif j == _SPICE:
                    chosen.append(_SPICE)
                elif j == _SILK:
                    chosen.append(_SILK)
                elif j == _SILVER:
                    chosen.append(_SILVER)
                elif j == _GOLD:
                    chosen.append(_GOLD)
                elif j == _DIAMOND:
                    chosen.append(_DIAMOND)

            if (len(replacable) == len(chosen)) and len(replacable) >= 2:
                print(
                    "Trading cards", chosen, " from the marketplace with ", replacable
                )
                self.take_goods(chosen, player, replacable)

        # updating player herd number
        player.herd = player.cards[_CAMEL]

        # player will know opponent's number of camels
        if player.id == 0:
            # if p1 is playing
            self.p2.opp_camels = player.herd
        elif player.id == 1:
            # if p2 is playing
            self.p1.opp_camels = player.herd

    def get_masked_options(self, agent):
        if agent == "player_1":
            player = self.p1
        elif agent == "player_2":
            player = self.p2
        else:
            print("Error! Incorrect agent entered.")
            return

        options = []
        for i in range(0, 25469):
            options.append(1)  # all possible

        # take 1 goods
        # checking that player has less than 7 cards in his hand
        if player.hand <= 6:
            if _DIAMOND not in self.marketplace:
                options[0] = 0

            if _GOLD not in self.marketplace:
                options[1] = 0

            if _SILVER not in self.marketplace:
                options[2] = 0

            if _SILK not in self.marketplace:
                options[3] = 0

            if _SPICE not in self.marketplace:
                options[4] = 0

            if _LEATHER not in self.marketplace:
                options[5] = 0

        else:
            # if player has 7 cards in hand, they cannot take another card
            for i in range(0, 6):
                options[i] = 0

        mar_camel = 0
        # checking if there are Camel cards in the marketplace
        if _CAMEL not in self.marketplace:
            options[6] = 0
        else:
            # to be used for trading cards check
            for i in self.marketplace:
                if i == _CAMEL:
                    mar_camel = mar_camel + 1

        # checking if the player can sell any cards
        if player.cards[_DIAMOND] < 2:
            options[7] = 0

        if player.cards[_GOLD] < 2:
            options[8] = 0

        if player.cards[_SILVER] < 2:
            options[9] = 0

        if player.cards[_SILK] < 1:
            options[10] = 0

        if player.cards[_SPICE] < 1:
            options[11] = 0

        if player.cards[_LEATHER] < 1:
            options[12] = 0

        # if player doesn't have at least 2 cards or marketplace has 4 or more camels, all trading options are impossible
        if ((player.hand + player.herd) < 2) or mar_camel >= 4:
            # ("All trading card options impossible")
            for i in range(13, 25469):
                options[i] = 0
        else:

            mar_leather = 0
            mar_spice = 0
            mar_silk = 0
            mar_silver = 0
            mar_gold = 0
            mar_diamond = 0

            for i in self.marketplace:
                if i == _LEATHER:
                    mar_leather = mar_leather + 1
                elif i == _SPICE:
                    mar_spice = mar_spice + 1
                elif i == _SILK:
                    mar_silk = mar_silk + 1
                elif i == _SILVER:
                    mar_silver = mar_silver + 1
                elif i == _GOLD:
                    mar_gold = mar_gold + 1
                elif i == _DIAMOND:
                    mar_diamond = mar_diamond + 1

            # trading options
            for i in self.Q_trade:
                diamond = 0
                gold = 0
                silver = 0
                silk = 0
                spice = 0
                leather = 0
                camel = 0

                for j in self.Q_trade[i][0]:  # in_hand options

                    if j == _CAMEL:
                        camel = camel + 1
                    elif j == _LEATHER:
                        leather = leather + 1
                    elif j == _SPICE:
                        spice = spice + 1
                    elif j == _SILK:
                        silk = silk + 1
                    elif j == _SILVER:
                        silver = silver + 1
                    elif j == _GOLD:
                        gold = gold + 1
                    elif j == _DIAMOND:
                        diamond = diamond + 1

                if camel > 0 and (
                    camel > player.cards[_CAMEL] or camel > (7 - player.hand)
                ):
                    options[i] = 0  # not possible

                elif leather > 0 and leather > player.cards[_LEATHER]:
                    options[i] = 0  # not possible

                elif spice > 0 and spice > player.cards[_SPICE]:
                    options[i] = 0  # not possible

                elif silk > 0 and silk > player.cards[_SILK]:
                    options[i] = 0  # not possible

                elif silver > 0 and silver > player.cards[_SILVER]:
                    options[i] = 0  # not possible

                elif gold > 0 and gold > player.cards[_GOLD]:
                    options[i] = 0  # not possible

                elif diamond > 0 and diamond > player.cards[_DIAMOND]:
                    options[i] = 0  # not possible

                if options[i] == 1:  # if option is still possible
                    diamond_mar = 0
                    gold_mar = 0
                    silver_mar = 0
                    silk_mar = 0
                    spice_mar = 0
                    leather_mar = 0

                    for k in self.Q_trade[i][1]:  # looping through marketplace options
                        if k == _LEATHER:
                            leather_mar = leather_mar + 1
                        elif k == _SPICE:
                            spice_mar = spice_mar + 1
                        elif k == _SILK:
                            silk_mar = silk_mar + 1
                        elif k == _SILVER:
                            silver_mar = silver_mar + 1
                        elif k == _GOLD:
                            gold_mar = gold_mar + 1
                        elif k == _DIAMOND:
                            diamond_mar = diamond_mar + 1

                    if leather_mar > 0 and leather_mar > mar_leather:
                        options[i] = 0  # not possible

                    elif spice_mar > 0 and spice_mar > mar_spice:
                        options[i] = 0  # not possible

                    elif silk_mar > 0 and silk_mar > mar_silk:
                        options[i] = 0  # not possible

                    elif silver_mar > 0 and silver_mar > mar_silver:
                        options[i] = 0  # not possible

                    elif gold_mar > 0 and gold_mar > mar_gold:
                        options[i] = 0  # not possible

                    elif diamond_mar > 0 and diamond_mar > mar_diamond:
                        options[i] = 0  # not possible

        masked = np.array(options, dtype=np.int8)

        return masked


j = Jaipur()
