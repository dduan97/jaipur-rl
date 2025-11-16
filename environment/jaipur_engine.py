import dataclasses
from strenum import StrEnum
import itertools
from collections import Counter
import copy

import numpy as np


class GoodType(StrEnum):
    DIAMOND = "DIAMOND"
    GOLD = "GOLD"
    SILVER = "SILVER"
    SILK = "SILK"
    SPICE = "SPICE"
    LEATHER = "LEATHER"
    CAMEL = "CAMEL"

    def is_expensive_good(self):
        return self in {GoodType.DIAMOND, GoodType.GOLD, GoodType.SILVER}

    def __repr__(self):
        return self.value


class ActionType(StrEnum):
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
    def __init__(self, name: str):
        self.name = name
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

    def __repr__(self):
        template = f"""PLAYER {self.name}:
CARDS: {self.cards}, BONUS: {self.bonus}, OPP_POINTS: {self.opp_points}, OPP_HERD_SIZE: {self.opp_herd_size}, TOKENS: {self.tokens}"""
        return template


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
            GoodType.DIAMOND: 6,
            GoodType.GOLD: 6,
            GoodType.SILVER: 6,
            GoodType.SILK: 8,
            GoodType.SPICE: 8,
            GoodType.LEATHER: 10,
            GoodType.CAMEL: 11,
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

    def __repr__(self):
        return f"""NUM CARDS IN DECK: {len(self.cards)}"""


class TokenDeck:
    def __init__(self):
        # 60 tokens

        # 38 goods
        # 18 bonus
        # 1 camel
        # 3 soe
        self.goods_type_tokens = {
            GoodType.DIAMOND: [7, 7, 5, 5, 5],
            GoodType.GOLD: [6, 6, 5, 5, 5],
            GoodType.SILVER: [5, 5, 5, 5, 5],
            GoodType.SILK: [5, 3, 3, 2, 2, 1, 1],
            GoodType.SPICE: [5, 3, 3, 2, 2, 1, 1],
            GoodType.LEATHER: [4, 3, 2, 1, 1, 1, 1, 1, 1],
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
        return [g for g, v in self.goods_type_tokens.items() if not v]

    def __repr__(self):
        num_goods_tokens = sum(len(v) for v in self.goods_type_tokens.values())
        num_bonus_tokens = sum(len(v) for v in self.bonus_tokens.values())
        return f"""NUM GOODS TOKENS: {num_goods_tokens}
NUM BONUS TOKENS: {num_bonus_tokens}"""


@dataclasses.dataclass
class JaipurGameState:
    goods: GoodsDeck
    tokens: TokenDeck
    marketplace: dict[GoodType, int]
    discard: dict[GoodType, int]
    player_names: list[str]
    players: dict[str, Player]

    def __repr__(self):
        goods_str = str(self.goods).replace("\n", "\n\t")
        tokens_str = str(self.tokens).replace("\n", "\n\t")
        marketplace_str = str(self.marketplace).replace("\n", "\n\t")
        players_str = "\n\t".join(
            str(v).replace("\n", "\n\t") for v in self.players.values()
        )
        return f"""GOODS:
\t{goods_str}
TOKENS:
\t{tokens_str}
MARKETPLACE:
\t{marketplace_str}
PLAYERS:
\t{players_str}
"""


class JaipurEngine:
    def __init__(self, player_names: list[str]):

        self.game_state = self.init_game_state(player_names)
        self.all_actions = self.get_all_actions()

    def init_game_state(self, player_names: list[str]) -> JaipurGameState:
        goods = GoodsDeck()
        tokens = TokenDeck()

        players = {}
        for name in player_names:
            player = Player(name)
            player.reset()
            players[name] = player

        marketplace = {g: 0 for g in GoodType}
        discard = {g: 0 for g in GoodType}

        # adding 3 camel cards to the marketplace
        for _ in range(3):
            marketplace[GoodType.CAMEL] += 1
            goods.deal_camel()

        # shuffle cards
        goods.shuffle()

        # adding 2 cards from shuffled cards to the marketplace
        for _ in range(2):
            good = goods.deal()
            # This won't happen but pytype needs it
            if good is not None:
                marketplace[good] += 1

        # giving out 5 cards to each player
        for player in players.values():
            for _ in range(5):
                # pop card from deck
                c = goods.deal()
                player.add_good(c)

        # setting opponent's herd size
        players[player_names[0]].opp_herd_size = players[player_names[1]].herd_size()
        players[player_names[1]].opp_herd_size = players[player_names[0]].herd_size()

        return JaipurGameState(
            goods=goods,
            tokens=tokens,
            marketplace=marketplace,
            discard=discard,
            player_names=player_names,
            players=players,
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
                    actions.append(
                        JaipurAction(
                            action_type=ActionType.TRADE_WITH_MARKETPLACE,
                            trade_from_hand=dict(Counter(in_hand_goods)),
                            trade_from_marketplace=dict(Counter(marketplace_goods)),
                        )
                    )

        # And then all the sell actions
        for num_to_sell in range(1, 7):
            for good_type in good_types_no_camel:
                # Only 6 diamond, gold, and silver
                if num_to_sell == 7 and good_type.is_expensive_good():
                    continue

                actions.append(
                    JaipurAction(
                        action_type=ActionType.SELL_CARDS,
                        sell_from_hand={good_type: num_to_sell},
                    )
                )

        return actions

    def is_valid(self, player_name: str, action: JaipurAction) -> bool:
        if player_name not in self.game_state.players:
            raise ValueError("Unknown player", player_name)
        player = self.game_state.players[player_name]

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
            )

        elif action.action_type == ActionType.TRADE_WITH_MARKETPLACE:
            # Shouldn't happen but pytype
            if action.trade_from_hand is None or action.trade_from_marketplace is None:
                raise ValueError("Trade action but no cards specified")
            # Shouldn't happen
            if GoodType.CAMEL in action.trade_from_marketplace:
                raise ValueError(
                    "Trade is invalid, can't have action type TRADE where we're taking camels from the marketplace"
                )
            # If the marketplace items are all present
            for good_type, count in action.trade_from_hand.items():
                if player.cards.get(good_type, 0) < count:
                    return False
            for good_type, count in action.trade_from_marketplace.items():
                if self.game_state.marketplace.get(good_type, 0) < count:
                    return False
            # Make sure we don't end up with too many cards, since we can trade from both hand and herd
            goods_leaving_hand = sum(v for k, v in action.trade_from_hand.items() if k != GoodType.CAMEL)
            net_new_hand = sum(action.trade_from_marketplace.values()) - goods_leaving_hand
            if player.hand_size() + net_new_hand > 7:
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

    def perform_action(self, player_name: str, action_idx: int):
        if player_name not in self.game_state.players:
            raise ValueError("Unknown player", player_name)

        player = self.game_state.players[player_name]
        action = self.all_actions[action_idx]
        # print("Processing action", action, "for", player_name)

        player_before_action = copy.deepcopy(player)

        if not self.is_valid(player_name, action):
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
                    self.game_state.marketplace[good_type] += 1

            for good_type, count in action.trade_from_marketplace.items():
                for _ in range(count):
                    self.game_state.marketplace[good_type] -= 1
                    player.add_good(good_type)

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
                # Selling more than 5 gives you the 5 bonus
                bonus_count = min(count, 5)
                bonus_token = self.game_state.tokens.take_bonus_tokens(bonus_count)
                if bonus_token:
                    tokens.append(bonus_token)
            player.add_tokens(tokens)

        else:
            raise ValueError("Unknown action type: %s", action.action_type)

        # Sanity check
        if player.hand_size() > 7:
            print('PLAYER BEFORE ACTION', player_before_action)
            print('ACTION', action)
            print('PLAYER AFTER ACTION', player)
            raise ValueError('Player has too many cards!')
            

    def finalize_round(self):
        """Call this after a round is finished. This will add the camel token."""
        p1 = self.game_state.players[self.game_state.player_names[0]]
        p2 = self.game_state.players[self.game_state.player_names[1]]
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

    def compute_score(self, player_name: str) -> int:
        player = self.game_state.players[player_name]
        return sum(player.tokens)

    def get_masked_options(self, player_name: str):
        """Given a player, return a the masked set of options that are available"""
        is_valid_for_player = [
            self.is_valid(player_name, action) for action in self.all_actions
        ]
        return is_valid_for_player

    def get_observation(self, player_name: str):
        if player_name not in self.game_state.players:
            raise ValueError("Invalid player name", player_name)
        player = self.game_state.players[player_name]
        # hmmm not ideal
        opponent = [p for k, p in self.game_state.players.items() if k != player_name][
            0
        ]

        features = []
        # First the marketplace features
        for good_type in GoodType:
            features.append(self.game_state.marketplace[good_type])

        # Then the player's hand
        for good_type in GoodType:
            if good_type == GoodType.CAMEL:
                continue
            features.append(player.cards[good_type])

        # Then the opponent hand size
        features.append(opponent.hand_size())

        # Then the herd sizes
        features.append(player.herd_size())
        features.append(opponent.herd_size())

        # And the remaining goods tokens
        for good_type in GoodType:
            if good_type == GoodType.CAMEL:
                continue
            features.append(len(self.game_state.tokens.goods_type_tokens[good_type]))

        # And the bonus tokens
        for bonus_size in [3, 4, 5]:
            features.append(len(self.game_state.tokens.bonus_tokens[bonus_size]))

        # And whether the deck has cards left:
        features.append(1 if self.game_state.goods.cards else 0)

        # And the cards in the discard:
        for good_type in GoodType:
            if good_type == GoodType.CAMEL:
                continue
            features.append(self.game_state.discard[good_type])

        # And the scores for each player
        features.append(sum(player.tokens))
        features.append(sum(opponent.tokens))

        assert len(features) == 34, f"Got {len(features)} features"
        return features
