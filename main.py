import math
from decimal import Decimal
from typing import Dict, Tuple, Set, Optional, List

# Constants for Uniswap V3
MIN_TICK = -887272
MAX_TICK = 887272
Q96 = 2**96
ZERO_ADDR = "0x0000000000000000000000000000000000000000"

class UniswapV3PoolState:
    def __init__(self, fee: int, tick_spacing: int, token0: str, token1: str):
        # Pool parameters
        self.fee = fee
        self.tick_spacing = tick_spacing
        self.token0 = token0
        self.token1 = token1
        
        # Current state
        self.sqrt_price_x96 = 0
        self.current_tick = 0
        self.current_liquidity = 0
        
        # Tick tracking
        self.tick_bitmap: Dict[int, int] = {}  # wordPos -> bitmap
        self.ticks_liquidity_net: Dict[int, int] = {}  # tick -> net liquidity change
        
        # Position tracking
        self.positions: Dict[Tuple[str, int, int], int] = {}  # (owner, tickLower, tickUpper) -> liquidity
        
        # NFT position tracking
        self.nft_positions: Dict[int, Tuple[str, int, int, int]] = {}  # tokenId -> (owner, tickLower, tickUpper, liquidity)
    

    def is_tick_initialized(self, tick: int) -> bool:
        """Check if a tick is initialized in the bitmap."""
        word_pos = tick >> 8
        bit_pos = tick % 256
        
        if word_pos not in self.tick_bitmap:
            return False
        
        return (self.tick_bitmap[word_pos] & (1 << bit_pos)) != 0
    
    def set_tick_initialized(self, tick: int, initialized: bool):
        """Set or unset a tick as initialized in the bitmap."""
        word_pos = tick >> 8
        bit_pos = tick % 256
        
        if word_pos not in self.tick_bitmap:
            self.tick_bitmap[word_pos] = 0
        
        if initialized:
            self.tick_bitmap[word_pos] |= (1 << bit_pos)
        else:
            self.tick_bitmap[word_pos] &= ~(1 << bit_pos)
    
    def get_next_initialized_tick_within_word(self, tick: int, lte: bool) -> Optional[int]:
        """
        Find the next initialized tick in the same word.
        lte=True searches for tick ≤ current tick (backwards)
        lte=False searches for tick > current tick (forwards)
        """
        word_pos = tick >> 8
        if word_pos not in self.tick_bitmap:
            return None
            
        word = self.tick_bitmap[word_pos]
        
        # No initialized ticks in this word
        if word == 0:
            return None
            
        # Get the position within the word
        bit_pos = tick % 256
        
        if lte:
            # Mask for all bits at or below the current position
            mask = (1 << (bit_pos + 1)) - 1
            masked_word = word & mask
            
            if masked_word == 0:
                return None
                
            # rightmost bit position
            result_bit_pos = masked_word.bit_length() - 1
        else:
            # Mask for all bits strictly above the current position
            mask = ~((1 << (bit_pos + 1)) - 1) & ((1 << 256) - 1)
            masked_word = word & mask
            
            if masked_word == 0:
                return None
                
            # leftmost bit position
            result_bit_pos = (masked_word & -masked_word).bit_length() - 1
            
        return (word_pos << 8) + result_bit_pos
    
    def next_initialized_tick(self, tick: int, lte: bool) -> int:
        """
        Find the next initialized tick.
        lte=True searches for tick ≤ current tick (backwards)
        lte=False searches for tick > current tick (forwards)
        """
        if lte:
            # Round down to the nearest tick spacing multiple
            tick = math.floor(tick / self.tick_spacing) * self.tick_spacing
            
            word_pos = tick >> 8
            # First try to find in the current word
            next_tick = self.get_next_initialized_tick_within_word(tick, True)
            if next_tick is not None:
                return next_tick
            
            # Search backwards through words
            word_pos -= 1
            while word_pos >= MIN_TICK >> 8:
                if word_pos in self.tick_bitmap and self.tick_bitmap[word_pos] > 0:
                    # Find the most significant bit
                    word = self.tick_bitmap[word_pos]
                    bit_pos = word.bit_length() - 1
                    return (word_pos << 8) + bit_pos
                word_pos -= 1
            
            return MIN_TICK
        else:
            # Round up to the nearest tick spacing multiple
            tick = math.ceil(tick / self.tick_spacing) * self.tick_spacing
            
            word_pos = tick >> 8
            # First try to find in the current word
            next_tick = self.get_next_initialized_tick_within_word(tick, False)
            if next_tick is not None:
                return next_tick
            
            # Search forwards through words
            word_pos += 1
            while word_pos <= MAX_TICK >> 8:
                if word_pos in self.tick_bitmap and self.tick_bitmap[word_pos] > 0:
                    # Find the least significant bit
                    word = self.tick_bitmap[word_pos]
                    bit_pos = (word & -word).bit_length() - 1
                    return (word_pos << 8) + bit_pos
                word_pos += 1
            
            return MAX_TICK

    def get_sqrt_ratio_at_tick(self, tick: int) -> int:
        """Calculate the square root price at a given tick (follows TickMath.sol)."""
        tick = max(MIN_TICK, min(tick, MAX_TICK))
        
        abs_tick = abs(tick)
        
        # For precision, we use Decimal for the intermediate calculations
        ratio = Decimal(1.0001) ** (Decimal(abs_tick) / 2)
        
        if tick < 0:
            ratio = Decimal(1) / ratio
        
        return int(ratio * Decimal(2**96))

    def get_tick_at_sqrt_ratio(self, sqrt_ratio_x96: int) -> int:
        """Calculate the tick for a given square root price (follows TickMath.sol)."""
        # Convert sqrt_ratio_x96 to decimal
        sqrt_ratio = Decimal(sqrt_ratio_x96) / Decimal(2**96)
        
        # Calculate log base 1.0001 of the ratio squared
        price = sqrt_ratio ** 2
        log_price = math.log(float(price), 1.0001)
        
        # Round to the nearest integer
        tick = int(log_price)
        
        # Adjust tick if needed
        if (Decimal(1.0001) ** Decimal(tick)) > price and tick > MIN_TICK:
            tick -= 1
            
        return max(MIN_TICK, min(tick, MAX_TICK))

    def calculate_amount0(self, sqrt_ratio_lower_x96: int, sqrt_ratio_upper_x96: int, liquidity: int) -> int:
        """Calculate amount of token0 required for given liquidity (follows LiquidityAmounts.sol)."""
        if sqrt_ratio_lower_x96 >= sqrt_ratio_upper_x96:
            raise ValueError("sqrt_ratio_lower_x96 must be less than sqrt_ratio_upper_x96")
        
        # Calculate using the formula from LiquidityAmounts.sol
        # amount0 = liquidity * (1/sqrt_lower - 1/sqrt_upper) * 2^96
        amount0 = (liquidity * Q96 * (sqrt_ratio_upper_x96 - sqrt_ratio_lower_x96)) // (sqrt_ratio_lower_x96 * sqrt_ratio_upper_x96)
        
        return amount0
    
    def calculate_amount1(self, sqrt_ratio_lower_x96: int, sqrt_ratio_upper_x96: int, liquidity: int) -> int:
        """Calculate amount of token1 required for given liquidity (follows LiquidityAmounts.sol)."""
        if sqrt_ratio_lower_x96 >= sqrt_ratio_upper_x96:
            raise ValueError("sqrt_ratio_lower_x96 must be less than sqrt_ratio_upper_x96")
        
        # Calculate using the formula from LiquidityAmounts.sol
        # amount1 = liquidity * (sqrt_upper - sqrt_lower)
        amount1 = (liquidity * (sqrt_ratio_upper_x96 - sqrt_ratio_lower_x96)) // Q96
        
        return amount1


    def calculate_liquidity_for_amounts(
        self, 
        sqrt_ratio_x96: int,
        sqrt_ratio_lower_x96: int, 
        sqrt_ratio_upper_x96: int,
        amount0: int,
        amount1: int
    ) -> int:
        """Calculate liquidity from token amounts (follows LiquidityAmounts.sol)."""
        if sqrt_ratio_lower_x96 >= sqrt_ratio_upper_x96:
            raise ValueError("sqrt_ratio_lower_x96 must be less than sqrt_ratio_upper_x96")
        
        liquidity = 0
        
        # Current price is below the range, only token0 is used
        if sqrt_ratio_x96 <= sqrt_ratio_lower_x96:
            liquidity = self.calculate_liquidity_for_amount0(
                sqrt_ratio_lower_x96, 
                sqrt_ratio_upper_x96, 
                amount0
            )
        # Current price is above the range, only token1 is used
        elif sqrt_ratio_x96 >= sqrt_ratio_upper_x96:
            liquidity = self.calculate_liquidity_for_amount1(
                sqrt_ratio_lower_x96, 
                sqrt_ratio_upper_x96, 
                amount1
            )
        # Current price is in the range, both tokens are used
        else:
            # Calculate liquidity for each token
            liquidity0 = self.calculate_liquidity_for_amount0(
                sqrt_ratio_x96, 
                sqrt_ratio_upper_x96, 
                amount0
            )
            liquidity1 = self.calculate_liquidity_for_amount1(
                sqrt_ratio_lower_x96, 
                sqrt_ratio_x96, 
                amount1
            )
            # Use the smaller liquidity to ensure we don't exceed either token amount
            liquidity = min(liquidity0, liquidity1)
            
            # Print debug info
            print(f"Debug: liquidity0={liquidity0}, liquidity1={liquidity1}")
            print(f"Debug: Using min liquidity={liquidity}")
        
        return liquidity
    
    def calculate_liquidity_for_amount0(self, sqrt_ratio_a_x96: int, sqrt_ratio_b_x96: int, amount0: int) -> int:
        """Calculate liquidity from token0 amount only."""
        # Ensure lower sqrt ratio is used as sqrt_ratio_a_x96
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96
        
        # Use Decimal for high precision
        numerator = Decimal(amount0) * Decimal(sqrt_ratio_a_x96) * Decimal(sqrt_ratio_b_x96)
        denominator = Decimal(sqrt_ratio_b_x96 - sqrt_ratio_a_x96) * Decimal(Q96)
        
        return int(numerator / denominator)

    def calculate_liquidity_for_amount1(self, sqrt_ratio_a_x96: int, sqrt_ratio_b_x96: int, amount1: int) -> int:
        """Calculate liquidity from token1 amount only."""
        # Ensure lower sqrt ratio is used as sqrt_ratio_a_x96
        if sqrt_ratio_a_x96 > sqrt_ratio_b_x96:
            sqrt_ratio_a_x96, sqrt_ratio_b_x96 = sqrt_ratio_b_x96, sqrt_ratio_a_x96
        
        # Use Decimal for high precision
        return int(Decimal(amount1) * Decimal(Q96) / Decimal(sqrt_ratio_b_x96 - sqrt_ratio_a_x96))
    
    def add_liquidity(
        self,
        owner: str,
        tick_lower: int,
        tick_upper: int,
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int = 0,
        amount1_min: int = 0
    ) -> Tuple[int, int, int]:
        """
        Add liquidity following LiquidityManagement.sol logic.
        
        Returns:
            (liquidity, amount0, amount1)
        """
        # Validate tick range
        assert tick_lower < tick_upper, "IL"  # Invalid lower tick
        assert tick_lower >= MIN_TICK, "TLM"  # Tick lower too low
        assert tick_upper <= MAX_TICK, "TUM"  # Tick upper too high
        assert tick_lower % self.tick_spacing == 0, "TLM"  # Tick lower not spacing multiple
        assert tick_upper % self.tick_spacing == 0, "TUM"  # Tick upper not spacing multiple
        
        # Get sqrt price at ticks
        sqrt_price_lower_x96 = self.get_sqrt_ratio_at_tick(tick_lower)
        sqrt_price_upper_x96 = self.get_sqrt_ratio_at_tick(tick_upper)
        
        # Calculate liquidity from desired amounts
        liquidity = self.calculate_liquidity_for_amounts(
            self.sqrt_price_x96,
            sqrt_price_lower_x96,
            sqrt_price_upper_x96,
            amount0_desired,
            amount1_desired
        )
        
        # No liquidity to add
        if liquidity <= 0:
            raise ValueError("Insufficient liquidity")
        
        # Calculate actual amounts based on liquidity
        amount0 = 0
        amount1 = 0
        
        if self.sqrt_price_x96 <= sqrt_price_lower_x96:
            # Current price is below the range
            amount0 = self.calculate_amount0(sqrt_price_lower_x96, sqrt_price_upper_x96, liquidity)
            amount1 = 0
        elif self.sqrt_price_x96 < sqrt_price_upper_x96:
            # Current price is within the range
            amount0 = self.calculate_amount0(self.sqrt_price_x96, sqrt_price_upper_x96, liquidity)
            amount1 = self.calculate_amount1(sqrt_price_lower_x96, self.sqrt_price_x96, liquidity)
        else:
            # Current price is above the range
            amount0 = 0
            amount1 = self.calculate_amount1(sqrt_price_lower_x96, sqrt_price_upper_x96, liquidity)
        
        # Slippage check
        if amount0 > amount0_desired:
            print(f"Warning: amount0 {amount0} > desired {amount0_desired}")
            amount0 = amount0_desired  # Cap at the desired amount
        
        if amount1 > amount1_desired:
            print(f"Warning: amount1 {amount1} > desired {amount1_desired}")
            amount1 = amount1_desired  # Cap at the desired amount
            
        if amount0 < amount0_min:
            raise ValueError(f"Insufficient token0: {amount0} < {amount0_min}")
        
        if amount1 < amount1_min:
            raise ValueError(f"Insufficient token1: {amount1} < {amount1_min}")
        
        # Add the liquidity to the pool
        self.process_mint(owner, tick_lower, tick_upper, liquidity)
        
        return liquidity, amount0, amount1
    
    def process_mint(self, owner: str, tick_lower: int, tick_upper: int, amount: int):
        """Process a mint (add liquidity) event."""
        # Ensure ticks are divisible by tick spacing
        assert tick_lower % self.tick_spacing == 0, "Lower tick must be divisible by tick spacing"
        assert tick_upper % self.tick_spacing == 0, "Upper tick must be divisible by tick spacing"
        assert tick_lower < tick_upper, "Lower tick must be less than upper tick"
        
        # Create position key
        pos_key = (owner, tick_lower, tick_upper)
        
        # Update position
        if pos_key in self.positions:
            self.positions[pos_key] += amount
        else:
            self.positions[pos_key] = amount
        
        # Update net liquidity at the ticks
        if tick_lower in self.ticks_liquidity_net:
            self.ticks_liquidity_net[tick_lower] += amount
        else:
            self.ticks_liquidity_net[tick_lower] = amount
            
        if tick_upper in self.ticks_liquidity_net:
            self.ticks_liquidity_net[tick_upper] -= amount
        else:
            self.ticks_liquidity_net[tick_upper] = -amount
        
        # If position is in current tick range, update current liquidity
        if self.current_tick >= tick_lower and self.current_tick < tick_upper:
            self.current_liquidity += amount
        
        # Set ticks as initialized
        self.set_tick_initialized(tick_lower, True)
        self.set_tick_initialized(tick_upper, True)
    
    def process_burn(self, owner: str, tick_lower: int, tick_upper: int, amount: int):
        """Process a burn (remove liquidity) event."""
        # Create position key
        pos_key = (owner, tick_lower, tick_upper)
        
        # Check position exists
        assert pos_key in self.positions, "Position does not exist"
        assert self.positions[pos_key] >= amount, "Not enough liquidity to burn"
        
        # Update position
        self.positions[pos_key] -= amount
        if self.positions[pos_key] == 0:
            del self.positions[pos_key]
        
        # Update net liquidity at the ticks
        self.ticks_liquidity_net[tick_lower] -= amount
        self.ticks_liquidity_net[tick_upper] += amount
        
        # If position is in current tick range, update current liquidity
        if self.current_tick >= tick_lower and self.current_tick < tick_upper:
            self.current_liquidity -= amount
        
        # Check if ticks should be uninitialized
        if self.ticks_liquidity_net[tick_lower] == 0:
            self.set_tick_initialized(tick_lower, False)
            del self.ticks_liquidity_net[tick_lower]
            
        if self.ticks_liquidity_net[tick_upper] == 0:
            self.set_tick_initialized(tick_upper, False)
            del self.ticks_liquidity_net[tick_upper]
    
    def process_swap(self, sqrt_price_x96: int, tick: int, liquidity: int):
        """Process a swap event by updating current state."""
        # Keep track of ticks we cross during this swap
        if tick != self.current_tick:
            # We've crossed at least one tick
            if tick > self.current_tick:
                # Moving up in price, enumerate all ticks we've crossed
                next_tick = self.next_initialized_tick(self.current_tick, False)
                while next_tick <= tick and next_tick < MAX_TICK:
                    # Cross this tick
                    if next_tick in self.ticks_liquidity_net:
                        self.current_liquidity += self.ticks_liquidity_net[next_tick]
                    next_tick = self.next_initialized_tick(next_tick, False)
            else:
                # Moving down in price, enumerate all ticks we've crossed
                next_tick = self.next_initialized_tick(self.current_tick, True)
                while next_tick > tick and next_tick > MIN_TICK:
                    # Cross this tick
                    if next_tick in self.ticks_liquidity_net:
                        self.current_liquidity -= self.ticks_liquidity_net[next_tick]
                    next_tick = self.next_initialized_tick(next_tick, True)
        
        # Update current state
        self.sqrt_price_x96 = sqrt_price_x96
        self.current_tick = tick
    
    def process_nft_mint(
        self, 
        token_id: int,
        owner: str, 
        tick_lower: int, 
        tick_upper: int, 
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int = 0,
        amount1_min: int = 0
    ) -> Tuple[int, int, int]:
        """
        Process a mint event from NonfungiblePositionManager.
        Creates a new NFT position and adds liquidity to the pool.
        
        Returns:
            (liquidity, amount0, amount1)
        """
        # Ensure token ID doesn't already exist
        assert token_id not in self.nft_positions, "Token ID already exists"
        
        # Add liquidity using the LiquidityManagement logic
        liquidity, amount0, amount1 = self.add_liquidity(
            owner, 
            tick_lower, 
            tick_upper, 
            amount0_desired, 
            amount1_desired, 
            amount0_min, 
            amount1_min
        )
        
        # Store the NFT position
        self.nft_positions[token_id] = (owner, tick_lower, tick_upper, liquidity)
        
        return liquidity, amount0, amount1
    
    def process_increase_liquidity(
        self, 
        token_id: int, 
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int = 0,
        amount1_min: int = 0
    ) -> Tuple[int, int, int]:
        """
        Process an IncreaseLiquidity event from NonfungiblePositionManager.
        Adds liquidity to an existing NFT position.
        
        Returns:
            (liquidity_added, amount0, amount1)
        """
        # Ensure token ID exists
        assert token_id in self.nft_positions, "Token ID does not exist"
        
        # Get the position details
        owner, tick_lower, tick_upper, current_liquidity = self.nft_positions[token_id]
        
        # Add liquidity using the LiquidityManagement logic
        liquidity_added, amount0, amount1 = self.add_liquidity(
            owner, 
            tick_lower, 
            tick_upper, 
            amount0_desired, 
            amount1_desired, 
            amount0_min, 
            amount1_min
        )
        
        # Update the NFT position
        new_liquidity = current_liquidity + liquidity_added
        self.nft_positions[token_id] = (owner, tick_lower, tick_upper, new_liquidity)
        
        return liquidity_added, amount0, amount1


    def process_decrease_liquidity(
        self, 
        token_id: int, 
        liquidity: int,
        amount0_min: int = 0,
        amount1_min: int = 0
    ) -> Tuple[int, int]:
        """
        Process a DecreaseLiquidity event from NonfungiblePositionManager.
        Removes liquidity from an existing NFT position.
        
        Returns:
            (amount0, amount1)
        """
        # Ensure token ID exists
        assert token_id in self.nft_positions, "Token ID does not exist"
        
        # Get the position details
        owner, tick_lower, tick_upper, current_liquidity = self.nft_positions[token_id]
        
        # Ensure there's enough liquidity to remove
        assert current_liquidity >= liquidity, "Not enough liquidity to remove"
        
        # Calculate the amounts that will be received
        # Get sqrt price at ticks
        sqrt_price_lower_x96 = self.get_sqrt_ratio_at_tick(tick_lower)
        sqrt_price_upper_x96 = self.get_sqrt_ratio_at_tick(tick_upper)
        
        # Calculate token amounts based on liquidity being removed
        amount0 = 0
        amount1 = 0
        
        if self.sqrt_price_x96 <= sqrt_price_lower_x96:
            # Current price is below the range
            amount0 = self.calculate_amount0(sqrt_price_lower_x96, sqrt_price_upper_x96, liquidity)
        elif self.sqrt_price_x96 < sqrt_price_upper_x96:
            # Current price is within the range
            amount0 = self.calculate_amount0(self.sqrt_price_x96, sqrt_price_upper_x96, liquidity)
            amount1 = self.calculate_amount1(sqrt_price_lower_x96, self.sqrt_price_x96, liquidity)
        else:
            # Current price is above the range
            amount1 = self.calculate_amount1(sqrt_price_lower_x96, sqrt_price_upper_x96, liquidity)
        
        # Slippage check
        assert amount0 >= amount0_min, "Insufficient token0"
        assert amount1 >= amount1_min, "Insufficient token1"
        
        # Update the NFT position
        new_liquidity = current_liquidity - liquidity
        if new_liquidity > 0:
            self.nft_positions[token_id] = (owner, tick_lower, tick_upper, new_liquidity)
        else:
            # If liquidity is now zero, we can remove the NFT position
            # Note: In reality, the NFT would still exist, just with zero liquidity
            del self.nft_positions[token_id]
        
        # Remove liquidity from the pool
        self.process_burn(owner, tick_lower, tick_upper, liquidity)
    
        return amount0, amount1

    
    def process_collect(
        self, 
        token_id: int, 
        recipient: str,
        amount0_max: int,
        amount1_max: int
    ) -> Tuple[int, int]:
        """
        Process a Collect event from NonfungiblePositionManager.
        Collects tokens that have been earned by the position.
        
        Returns:
            (amount0, amount1)
        """
        # Ensure token ID exists
        assert token_id in self.nft_positions, "Token ID does not exist"
        
        # Get the position details
        owner, tick_lower, tick_upper, liquidity = self.nft_positions[token_id]
        
        # In a full implementation, we would track fees per position
        # For simplicity, we return zeros (as if no fees were collected)
        # In a real implementation, you would calculate fees based on:
        # - Fee growth global
        # - Fee growth inside the range
        # - Position's fee growth checkpoint
        
        return 0, 0

    def process_transfer(self, token_id: int, from_address: str, to_address: str):
        """
        Process a Transfer event from NonfungiblePositionManager.
        Transfers ownership of an NFT position.
        """
        # Ensure token ID exists
        assert token_id in self.nft_positions, "Token ID does not exist"
        
        # Get the position details
        owner, tick_lower, tick_upper, liquidity = self.nft_positions[token_id]
        
        # Ensure current owner matches
        assert owner == from_address, "Sender is not the owner"
        
        # Update owner in NFT position
        self.nft_positions[token_id] = (to_address, tick_lower, tick_upper, liquidity)
        
        # Update the pool position
        # First remove from old owner
        self.process_burn(from_address, tick_lower, tick_upper, liquidity)
        # Then add to new owner
        self.process_mint(to_address, tick_lower, tick_upper, liquidity)



def reconstruct_pool_state_from_events(pool_address: str, events: List[dict]) -> UniswapV3PoolState:
    """
    Reconstruct the pool state from a list of events.
    
    Args:
        pool_address: Address of the pool
        events: List of events (Swap, Mint, Burn, IncreaseLiquidity, DecreaseLiquidity)
    
    Returns:
        Reconstructed pool state
    """
    # Initialize pool state
    # Note: Fee and tick spacing should be retrieved from pool creation event or contract
    pool_state = UniswapV3PoolState(fee=3000, tick_spacing=60)  # Example values for a 0.3% pool
    
    # Sort events by block number and transaction index
    sorted_events = sorted(
        events, 
        key=lambda e: (e["blockNumber"], e["transactionIndex"], e.get("logIndex", 0))
    )
    
    for event in sorted_events:
        event_name = event["event"]
        args = event["args"]
        
        if event_name == "Mint":
            pool_state.process_mint(
                owner=args["owner"],
                tick_lower=args["tickLower"],
                tick_upper=args["tickUpper"],
                amount=args["amount"]
            )
        
        elif event_name == "Burn":
            pool_state.process_burn(
                owner=args["owner"],
                tick_lower=args["tickLower"],
                tick_upper=args["tickUpper"],
                amount=args["amount"]
            )
        
        elif event_name == "Swap":
            pool_state.process_swap(
                sqrt_price_x96=args["sqrtPriceX96"],
                tick=args["tick"],
                liquidity=args["liquidity"]
            )
        
        elif event_name == "IncreaseLiquidity" and event["address"] != pool_address:
            # This is from NonfungiblePositionManager
            pool_state.process_increase_liquidity(
                token_id=args["tokenId"],
                liquidity_added=args["liquidity"]
            )
        
        elif event_name == "DecreaseLiquidity" and event["address"] != pool_address:
            # This is from NonfungiblePositionManager
            pool_state.process_decrease_liquidity(
                token_id=args["tokenId"],
                liquidity_removed=args["liquidity"]
            )
        
        # Additionally, we could process Transfer and Collect events
    
    return pool_state

def sqrt_price_to_tick(sqrt_price_x96: int) -> int:
    """Convert square root price to corresponding tick."""
    # Convert to decimal for more accurate math
    price = Decimal(sqrt_price_x96) ** 2 / Decimal(2**192)
    # Tick is log base 1.0001 of price
    tick = math.floor(math.log(float(price), 1.0001))
    return tick

def tick_to_sqrt_price(tick: int) -> int:
    """Convert tick to square root price."""
    # sqrt(1.0001^tick) * 2^96
    return int(Decimal(1.0001) ** (Decimal(tick) / 2) * Decimal(2**96))

def compute_swap_step(
    sqrt_price_x96: int,
    target_sqrt_price_x96: int,
    liquidity: int,
    amount_remaining: int,
    fee_pips: int,
    exact_in: bool,
    zero_for_one: bool  # True if swapping token0 for token1
) -> Tuple[int, int, int, int]:
    """
    Compute a single step of a swap.
    
    Returns:
        (sqrt_price_next_x96, amount_in, amount_out, fee_amount)
    """
    # Fee calculation
    fee_factor = 1000000 - fee_pips
    
    if exact_in:
        # Calculate the price impact with the remaining input amount
        if zero_for_one:
            # Calculate price impact using liquidity and input amount
            # When swapping from token0 to token1, the price decreases
            amount_remaining_less_fee = (amount_remaining * fee_factor) // 1000000
            sqrt_price_next_x96 = max(
                target_sqrt_price_x96,
                sqrt_price_x96 - (sqrt_price_x96 * amount_remaining_less_fee) // (liquidity * fee_factor + amount_remaining_less_fee)
            )
        else:
            # When swapping from token1 to token0, the price increases
            amount_remaining_less_fee = (amount_remaining * fee_factor) // 1000000
            sqrt_price_next_x96 = min(
                target_sqrt_price_x96,
                sqrt_price_x96 + (sqrt_price_x96 * amount_remaining_less_fee) // liquidity
            )
    else:
        # For exact output, use the target price if in range
        sqrt_price_next_x96 = target_sqrt_price_x96
    
    # Ensure sqrt_price_next_x96 doesn't exceed the target when needed
    if zero_for_one:
        sqrt_price_next_x96 = max(sqrt_price_next_x96, target_sqrt_price_x96)
    else:
        sqrt_price_next_x96 = min(sqrt_price_next_x96, target_sqrt_price_x96)
    
    # Calculate amounts based on price change
    if zero_for_one:
        # Calculate token0 in and token1 out
        amount_in = liquidity * abs(sqrt_price_x96 - sqrt_price_next_x96) // Q96
        amount_out = liquidity * abs(sqrt_price_x96 - sqrt_price_next_x96) // sqrt_price_x96 // sqrt_price_next_x96
    else:
        # Calculate token1 in and token0 out
        amount_in = liquidity * abs(sqrt_price_x96 - sqrt_price_next_x96) // Q96
        amount_out = liquidity * abs(sqrt_price_x96 - sqrt_price_next_x96) // sqrt_price_x96 // sqrt_price_next_x96
    
    # Apply fee
    if exact_in:
        fee_amount = amount_remaining - amount_in
    else:
        # For exact out, fee is based on the calculated input amount
        fee_amount = (amount_in * fee_pips) // (1000000 - fee_pips)
        amount_in += fee_amount
    
    return sqrt_price_next_x96, amount_in, amount_out, fee_amount

def quote_exact_input_single(
    pool_state: UniswapV3PoolState,
    zero_for_one: bool,  # True if swapping token0 for token1
    amount_in: int,
    sqrt_price_limit_x96: int = 0
) -> int:
    """
    Calculate the expected output amount for an exact input swap.
    
    Args:
        pool_state: Current state of the pool
        zero_for_one: Direction of the swap (token0 for token1 if True)
        amount_in: Exact input amount
        sqrt_price_limit_x96: Price limit for the swap (0 for no limit)
    
    Returns:
        Expected output amount
    """
    if amount_in <= 0:
        return 0
    
    # Initialize swap state
    sqrt_price_x96 = pool_state.sqrt_price_x96
    liquidity = pool_state.current_liquidity
    tick = pool_state.current_tick
    amount_remaining = amount_in
    amount_out = 0
    
    # Set price limit if not specified
    if sqrt_price_limit_x96 == 0:
        sqrt_price_limit_x96 = 1 + 1 if zero_for_one else 2**160 - 1
    
    # Ensure the price limit is in the right direction
    if zero_for_one:
        assert sqrt_price_limit_x96 < sqrt_price_x96, "Price limit already exceeded"
    else:
        assert sqrt_price_limit_x96 > sqrt_price_x96, "Price limit already exceeded"
    
    # Loop until all input amount is consumed or price limit is reached
    while amount_remaining > 0 and sqrt_price_x96 != sqrt_price_limit_x96:
        # Find the next initialized tick
        next_tick = pool_state.next_initialized_tick(tick, zero_for_one)
        
        # Calculate the target price for this step
        sqrt_price_next_x96 = tick_to_sqrt_price(next_tick)
        
        # Adjust based on price direction
        if zero_for_one:
            sqrt_price_next_x96 = min(sqrt_price_next_x96, sqrt_price_limit_x96)
        else:
            sqrt_price_next_x96 = max(sqrt_price_next_x96, sqrt_price_limit_x96)
        
        # Compute the swap step
        sqrt_price_x96, amount_in_step, amount_out_step, fee_amount = compute_swap_step(
            sqrt_price_x96,
            sqrt_price_next_x96,
            liquidity,
            amount_remaining,
            pool_state.fee,
            True,  # exact_in
            zero_for_one
        )
        
        # Update running amounts
        amount_remaining -= (amount_in_step + fee_amount)
        amount_out += amount_out_step
        
        # Check if we've hit a tick boundary
        if sqrt_price_x96 == sqrt_price_next_x96:
            # We've crossed a tick, update state
            tick = next_tick
            
            # Update liquidity if tick has net liquidity change
            if next_tick in pool_state.ticks_liquidity_net:
                delta = pool_state.ticks_liquidity_net[next_tick]
                if zero_for_one:
                    liquidity -= delta
                else:
                    liquidity += delta
        else:
            # We've reached the price limit before the next tick
            tick = sqrt_price_to_tick(sqrt_price_x96)
    
    return amount_out


# def example_usage():
#     # Initialize a pool state (0.3% fee pool with tick spacing of 60)
#     pool = UniswapV3PoolState(fee=3000, tick_spacing=60)
    
#     # Set initial state (sqrt price, tick, and liquidity would come from events)
#     pool.sqrt_price_x96 = 2**96  # 1:1 price ratio
#     pool.current_tick = 0
#     pool.current_liquidity = 0
    
#     # Process some mint events to build the pool state
#     # Add liquidity around the current price
#     pool.process_mint("user1", -120, 120, 1000000)  # Around ±1% range
#     pool.process_mint("user2", -600, 600, 2000000)  # Around ±6% range
#     pool.process_mint("user3", -1200, 1200, 3000000)  # Around ±12% range
    
#     # Add some concentrated liquidity in specific ranges
#     pool.process_mint("user4", 0, 600, 500000)  # Only for price increases
#     pool.process_mint("user5", -600, 0, 500000)  # Only for price decreases
    
#     # Simulate a swap - exact input of 1 ETH for USDC (token0 for token1)
#     amount_in = 10**18  # 1 ETH
#     amount_out = quote_exact_input_single(pool, True, amount_in)
#     print(f"Swapping {amount_in / 10**18} ETH for {amount_out / 10**6} USDC")
    
#     # Simulate a swap in the opposite direction
#     amount_in = 1000 * 10**6  # 1000 USDC
#     amount_out = quote_exact_input_single(pool, False, amount_in)
#     print(f"Swapping {amount_in / 10**6} USDC for {amount_out / 10**18} ETH")
    
#     # Now let's process a burn event
#     pool.process_burn("user1", -120, 120, 500000)  # Removing half of user1's liquidity
    
#     # Simulate the same swaps after liquidity removal
#     amount_in = 10**18  # 1 ETH
#     amount_out = quote_exact_input_single(pool, True, amount_in)
#     print(f"After burn: Swapping {amount_in / 10**18} ETH for {amount_out / 10**6} USDC")
    
#     # Process a swap event to update the state
#     new_sqrt_price_x96 = int(2**96 * 1.01)  # Price moves 1% higher
#     new_tick = sqrt_price_to_tick(new_sqrt_price_x96)
#     pool.process_swap(new_sqrt_price_x96, new_tick, pool.current_liquidity)
    
#     print(f"New price: {(new_sqrt_price_x96 / 2**96)**2}")
#     print(f"New tick: {new_tick}")
#     print(f"Current liquidity: {pool.current_liquidity}")

def example_nft_position_manager():
    """Demonstrate the usage of NFT position manager events with updated logic."""
    # Initialize a pool state (0.3% fee pool with tick spacing of 60)
    pool = UniswapV3PoolState(
        fee=3000, 
        tick_spacing=60,
        token0="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
        token1="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"   # WETH
    )
    
    # Set initial state (for a pool at ~2000 USDC/ETH)
    pool.sqrt_price_x96 = int(Decimal(2000).sqrt() * Decimal(2**96))
    pool.current_tick = pool.get_tick_at_sqrt_ratio(pool.sqrt_price_x96)
    pool.current_liquidity = 0
    
    print(f"Initial price: {(pool.sqrt_price_x96 / 2**96)**2}")
    print(f"Initial tick: {pool.current_tick}")
    
    # Create an NFT position around the current price
    # User wants to provide 10,000 USDC and 5 ETH
    amount0_desired = 10000 * 10**6  # 10,000 USDC (6 decimals)
    amount1_desired = 5 * 10**18     # 5 ETH (18 decimals)
    
    # Position range: +/- 10% from current price
    tick_lower = math.floor(pool.current_tick - 1000)  # ~10% down
    tick_lower = math.floor(tick_lower / pool.tick_spacing) * pool.tick_spacing
    
    tick_upper = math.ceil(pool.current_tick + 1000)   # ~10% up
    tick_upper = math.ceil(tick_upper / pool.tick_spacing) * pool.tick_spacing
    
    # *** FIX: Remove slippage protection for initial position ***
    # Mint an NFT position
    liquidity, amount0, amount1 = pool.process_nft_mint(
        token_id=1,
        owner="0xuser1",
        tick_lower=tick_lower,
        tick_upper=tick_upper,
        amount0_desired=amount0_desired,
        amount1_desired=amount1_desired,
        amount0_min=0,  # No minimum requirement
        amount1_min=0   # No minimum requirement
    )
    
    print("\nAfter mint NFT:")
    print(f"NFT Position: {pool.nft_positions[1]}")
    print(f"Liquidity: {liquidity}")
    print(f"USDC used: {amount0 / 10**6}")
    print(f"ETH used: {amount1 / 10**18}")
    print(f"Current pool liquidity: {pool.current_liquidity}")
    
    # Increase liquidity of the position
    amount0_desired = 5000 * 10**6  # 5,000 more USDC
    amount1_desired = 2.5 * 10**18  # 2.5 more ETH
    
    # *** FIX: Remove slippage protection for increasing liquidity ***
    liquidity_added, amount0, amount1 = pool.process_increase_liquidity(
        token_id=1,
        amount0_desired=amount0_desired,
        amount1_desired=amount1_desired,
        amount0_min=0,  # No minimum requirement
        amount1_min=0   # No minimum requirement
    )
    
    print("\nAfter increase liquidity:")
    print(f"NFT Position: {pool.nft_positions[1]}")
    print(f"Additional liquidity: {liquidity_added}")
    print(f"Additional USDC used: {amount0 / 10**6}")
    print(f"Additional ETH used: {amount1 / 10**18}")
    print(f"Current pool liquidity: {pool.current_liquidity}")
    
    # Decrease liquidity of the position by 30%
    owner, tick_lower, tick_upper, current_liquidity = pool.nft_positions[1]
    liquidity_to_remove = current_liquidity * 30 // 100
    
    amount0, amount1 = pool.process_decrease_liquidity(
        token_id=1,
        liquidity=liquidity_to_remove,
        amount0_min=0,  # No slippage protection for simplicity
        amount1_min=0   # No slippage protection for simplicity
    )
    
    print("\nAfter decrease liquidity:")
    print(f"NFT Position: {pool.nft_positions[1]}")
    print(f"Removed liquidity: {liquidity_to_remove}")
    print(f"USDC received: {amount0 / 10**6}")
    print(f"ETH received: {amount1 / 10**18}")
    print(f"Current pool liquidity: {pool.current_liquidity}")
    
    # Simulate price change (ETH goes up 5%)
    new_price_sqrt_x96 = int(Decimal(2000 * 1.05).sqrt() * Decimal(2**96))
    new_tick = pool.get_tick_at_sqrt_ratio(new_price_sqrt_x96)
    pool.process_swap(new_price_sqrt_x96, new_tick, pool.current_liquidity)
    
    print(f"\nNew price: {(pool.sqrt_price_x96 / 2**96)**2}")
    print(f"New tick: {pool.current_tick}")
    print(f"Current liquidity: {pool.current_liquidity}")
    
    # Decrease all remaining liquidity
    owner, tick_lower, tick_upper, current_liquidity = pool.nft_positions[1]
    
    amount0, amount1 = pool.process_decrease_liquidity(
        token_id=1,
        liquidity=current_liquidity,
        amount0_min=0,
        amount1_min=0
    )
    
    print("\nAfter removing all liquidity:")
    print(f"Token ID in nft_positions: {1 in pool.nft_positions}")
    print(f"USDC received: {amount0 / 10**6}")
    print(f"ETH received: {amount1 / 10**18}")
    print(f"Current pool liquidity: {pool.current_liquidity}")

if __name__ == "__main__":
    example_nft_position_manager()

