import dataclasses
import enum
import itertools
from collections import Counter

import numpy as np


class GoodType(enum.StrEnum):
    DIAMOND = "DIAMOND"
    GOLD = "GOLD"
    SILVER = "SILVER"
    SILK = "SILK"
    SPICE = "SPICE"
    LEATHER = "LEATHER"
    CAMEL = "CAMEL"

    def is_expensive_good(self):
        return self in {GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER}


class ActionType(enum.StrEnum):
    TRADE_WITH_MARKETPLACE = "TRADE_WITH_MARKETPLACE"
    TAKE_ONE_GOOD = "TAKE_ONE_GOOD"
    TAKE_CAMELS = "TAKE_CAMELS"
    SELL_CARDS = "SELL_CARDS"


@dataclasses.dataclass
class JaipurAction:
    action_type: ActionType

    # These two are only set for the trade types
    trade_from_hand: dict[GoodType, int] | None = None
    trade_from_marketplace: dict[GoodType, int] | None = None

    # These two are only set for the sell type
    sell_from_hand: dict[GoodType, int] | None = None

    # Only set for the take one good type
    take_one_good_type: GoodType | None = None


class Player:
    def __init__(self, p_id):
        self.id = p_id
        self.reset()

    def reset(self):
        self.cards = {
            GoodType.DIAMOND: 0,
            GoodType.GOLD: 0,
            GoodType.SILVER: 0,
            GoodType.SILK: 0,
            GoodType.SPICE: 0,
            GoodType.LEATHER: 0,
            GoodType.CAMEL: 0,
        }
        self.bonus = 0
        self.opp_points = 0
        self.opp_herd_size = 0
        self.tokens = []

    def add_good(self, good: GoodType):
        self.cards[good] += 1

    def remove_good(self, good: GoodType):
        if self.cards[good] <= 0:
            raise ValueError("Cannot remove good %s", good)
        self.cards[good] -= 1

    def hand_size(self):
        return sum(v for k, v in self.cards.items() if k != GoodType.CAMEL)

    def herd_size(self):
        return self.cards[GoodType.CAMEL]

    def add_tokens(self, tokens: list[int]):
        self.tokens.extend(tokens)


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
            GoodType.CAMEL: 11,
            GoodType.DIAMOND: 6,
            GoodType.GOLD: 6,
            GoodType.SILVER: 6,
            GoodType.SILK: 8,
            GoodType.SPICE: 8,
            GoodType.LEATHER: 10,
        }

        for goods_type, count in goods_counts.items():
            self.cards.extend([goods_type for _ in range(count)])

        self.cards.sort()

    def shuffle(self):
        np.random.shuffle(self.cards)

    def deal(self) -> GoodType | None:
        # taking one card from the deck of cards
        if len(self.cards) > 0:
            return self.cards.pop(0)
        return None

    def deal_camel(self):
        for i in range(len(self.cards)):
            if self.cards[i] == GoodType.CAMEL:
                return self.cards.pop(i)

    def num_remaining(self):
        return len(self.cards)


class TokenDeck:
    def __init__(self):
        # 60 tokens

        # 38 goods
        # 18 bonus
        # 1 camel
        # 3 soe
        self.goods_type_tokens = {
            GoodType.LEATHER: [4, 3, 2, 1, 1, 1, 1, 1, 1],
            GoodType.SPICE: [5, 3, 3, 2, 2, 1, 1],
            GoodType.SILK: [5, 3, 3, 2, 2, 1, 1],
            GoodType.SILVER: [5, 5, 5, 5, 5],
            GoodType.GOLD: [6, 6, 5, 5, 5],
            GoodType.DIAMOND: [7, 7, 5, 5, 5],
        }

        # 18 total bonus tokens
        self.bonus_tokens = {
            3: [1, 1, 2, 2, 2, 3, 3],
            4: [4, 4, 5, 5, 6, 6],
            5: [8, 8, 9, 10, 10],
        }

        # bonus tokens should be shuffled
        for tokens in self.bonus_tokens.values():
            np.random.shuffle(tokens)

        # self.soe = ["S1", "S2", "S3"] #3
        self.camel_token = 5
        # leather, spices, silk, silver, gold, diamonds, bonus, seals of excellence, camel

    def take_tokens(self, good_type: GoodType, count: int) -> list[int]:
        out = []
        for _ in range(count):
            if self.goods_type_tokens[good_type]:
                out.append(self.goods_type_tokens[good_type].pop(0))
        return out

    def take_bonus_tokens(self, count: int) -> int:
        if self.bonus_tokens[count]:
            return self.bonus_tokens[count].pop()
        return 0

    def empty_goods_tokens(self):
        return [g for g, v in self.goods_type_tokens if not v]


@dataclasses.dataclass
class JaipurGameState:
    goods: GoodsDeck
    tokens: TokenDeck
    marketplace: dict[GoodType, int]
    discard: dict[GoodType, int]

    player1: Player
    player2: Player


class JaipurEngine:
    def __init__(self):

        self.game_state = self.init_game_state()
        self.all_actions = self.get_all_actions()

    def init_game_state(self) -> JaipurGameState:
        goods = GoodsDeck()
        tokens = TokenDeck()

        p1 = Player(0)
        p2 = Player(1)
        p1.reset()
        p2.reset()

        marketplace = {g: 0 for g in GoodType}
        discard = {g: 0 for g in GoodType}

        # adding 3 camel cards to the marketplace
        for i in range(3):
            marketplace[GoodType.CAMEL] += 1
            goods.deal_camel()

        # shuffle cards
        goods.shuffle()

        # adding 2 cards from shuffled cards to the marketplace
        for i in range(2):
            good = goods.deal()
            # This won't happen but pytype needs it
            if good is not None:
                marketplace[good] += 1

        print("Marketplace:", marketplace)

        # giving out 5 cards to each player
        for p in p1, p2:
            for _ in range(5):
                # pop card from deck
                c = goods.deal()
                p.add_good(c)

        print("Player 1 cards: ", p1.cards)
        print("Player 2 cards: ", p2.cards)

        # setting opponent's herd size
        p1.opp_herd_size = p2.herd_size()
        p2.opp_herd_size = p1.herd_size()

        return JaipurGameState(
            goods=goods,
            tokens=tokens,
            marketplace=marketplace,
            discard=discard,
            player1=p1,
            player2=p2,
        )

    @classmethod
    def get_all_actions(cls):
        """Create a list of all possible actions."""
        actions = []

        # First: the take camel actions
        take_camel_action = JaipurAction(action_type=ActionType.TAKE_CAMELS)
        actions.append(take_camel_action)

        # Then the take 1 good action
        for good_type in GoodType:
            take_one_good_action = JaipurAction(
                action_type=ActionType.TAKE_ONE_GOOD, take_one_good_type=good_type
            )
            actions.append(take_one_good_action)

        all_good_types = [g for g in GoodType]
        good_types_no_camel = [g for g in GoodType if g != GoodType.CAMEL]

        # Now all the trade actions
        for num_to_trade in range(1, 6):
            # all possible combinations in hand
            for in_hand_goods in itertools.combinations_with_replacement(
                all_good_types, num_to_trade
            ):
                for marketplace_goods in itertools.combinations_with_replacement(
                    good_types_no_camel, num_to_trade
                ):
                    # Two sides should be disjoint
                    if set(in_hand_goods) & set(marketplace_goods):
                        continue
                    # Otherwise we have a valid action!
                    actions.append(JaipurAction(
                        action_type=ActionType.TRADE_WITH_MARKETPLACE,
                        trade_from_hand=dict(Counter(in_hand_goods)),
                        trade_from_marketplace=dict(Counter(marketplace_goods)),
                    ))

        # And then all the sell actions
        for num_to_sell in range(1, 7):
            for good_type in good_types_no_camel:
                # Only 6 diamond, gold, and silver
                if num_to_sell == 7 and good_type.is_expensive_good():
                    continue

                actions.append(JaipurAction(
                    action_type=ActionType.SELL_CARDS,
                    sell_from_hand={good_type: num_to_sell},
                ))

        print(f"Initialized {len(actions)} actions")
        return actions

    def is_valid(self, player: Player, action: JaipurAction) -> bool:
        if action.action_type == ActionType.TAKE_ONE_GOOD:
            # This is valid if the player has fewer than 7 cards
            if action.take_one_good_type is None:
                raise ValueError("Take one good action but no good set")
            return (
                player.hand_size() < 7
                and self.game_state.marketplace[action.take_one_good_type] >= 1
            )

        elif action.action_type == ActionType.TAKE_CAMELS:
            return (
                self.game_state.marketplace[GoodType.CAMEL] > 0
                and player.hand_size() + self.game_state.marketplace[GoodType.CAMEL]
                <= 7
            )

        elif action.action_type == ActionType.TRADE_WITH_MARKETPLACE:
            # Shouldn't happen but pytype
            if action.trade_from_hand is None or action.trade_from_marketplace is None:
                raise ValueError("Trade action but no cards specified")
            # If the marketplace items are all present
            for good_type, count in action.trade_from_hand.items():
                if player.cards.get(good_type, 0) < count:
                    return False
            for good_type, count in action.trade_from_marketplace.items():
                if self.game_state.marketplace.get(good_type, 0) < count:
                    return False
            return True

        elif action.action_type == ActionType.SELL_CARDS:
            if action.sell_from_hand is None:
                raise ValueError("Sell action but no cards specified")

            if len(action.sell_from_hand) > 1:
                raise ValueError("Can only sell one type of good at a time")

            good_type = list(action.sell_from_hand.keys())[0]
            count = action.sell_from_hand[good_type]

            if player.cards.get(good_type, 0) < count:
                return False
            # And minimum number of goods
            if good_type.is_expensive_good() and count < 2:
                return False
            return True

        else:
            raise ValueError("Unknown action type: %s", action.action_type)

    def perform_action(self, player: Player, action: JaipurAction):
        if not self.is_valid(player, action):
            raise ValueError("Invalid action for game state", action, self.game_state)

        if action.action_type == ActionType.TAKE_ONE_GOOD:
            if action.take_one_good_type is None:
                raise ValueError("Take one good action but no good set")
            player.add_good(action.take_one_good_type)
            self.game_state.marketplace[action.take_one_good_type] -= 1
            # And replce from the deck
            new_good = self.game_state.goods.deal()
            if new_good is not None:
                self.game_state.marketplace[new_good] += 1

        elif action.action_type == ActionType.TAKE_CAMELS:
            num_camels = self.game_state.marketplace[GoodType.CAMEL]
            self.game_state.marketplace[GoodType.CAMEL] = 0
            for _ in range(num_camels):
                player.add_good(GoodType.CAMEL)
                # And fill in the marketplace from the deck
                new_good = self.game_state.goods.deal()
                if new_good is not None:
                    self.game_state.marketplace[new_good] += 1

        elif action.action_type == ActionType.TRADE_WITH_MARKETPLACE:
            # Shouldn't happen but pytype
            if action.trade_from_hand is None or action.trade_from_marketplace is None:
                raise ValueError("Trade action but no cards specified")

            for good_type, count in action.trade_from_hand.items():
                for _ in range(count):
                    player.remove_good(good_type)

            for good_type, count in action.trade_from_marketplace.items():
                for _ in range(count):
                    self.game_state.marketplace[good_type] -= 1

        elif action.action_type == ActionType.SELL_CARDS:
            if action.sell_from_hand is None:
                raise ValueError("Sell action but no cards specified")

            # Remove the cards from the player's hand
            good_type = list(action.sell_from_hand.keys())[0]
            count = action.sell_from_hand[good_type]
            for _ in range(count):
                player.remove_good(good_type)

            # Add to discard pile
            self.game_state.discard[good_type] += count

            # Now do the tokens. Take as many tokens as discarded
            tokens = self.game_state.tokens.take_tokens(good_type, count)

            if count > 3:
                bonus_token = self.game_state.tokens.take_bonus_tokens(count)
                if bonus_token:
                    tokens.append(bonus_token)
            player.add_tokens(tokens)

        else:
            raise ValueError("Unknown action type: %s", action.action_type)

    def add_camel_token(self):
        p1 = self.game_state.player1
        p2 = self.game_state.player2
        # checking for which player had the most camels
        if p1.herd_size() != p2.herd_size():
            camel_bonus_player = p1 if p1.herd_size() > p2.herd_size() else p2
            camel_bonus_player.add_tokens([self.game_state.tokens.camel_token])

    def is_finished(self) -> bool:
        # 3 types of goods token are depleted
        if len(self.game_state.tokens.empty_goods_tokens()) >= 3:
            return True
        # or if there are no cards left in the draw pile when trying to fill the market
        if sum(self.game_state.marketplace.values()) < 5:
            return True
        return False

    def get_masked_options(self, agent: str):
        """Given a player, return a the masked set of options that are available"""
        if agent == "player_1":
            player = self.game_state.player1
        elif agent == "player_2":
            player = self.game_state.player2
        else:
            print("Error! Incorrect agent entered.")
            return

        is_valid_for_player = [self.is_valid(player, action) for action in self.all_actions]
        return is_valid_for_player


j = JaipurEngine()
