# 八、动规二

![在这里插入图片描述](https://img-blog.csdnimg.cn/205c74c5f89f497387f8957ae461effc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----


@[TOC](目录)



---
#### <font color=red face="微软雅黑">来源</font>
  [极客时间2021算法训练营](https://u.geekbang.org/lesson/194?article=419794&utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=u_nav_web)

作者:  李煜东


----
## 1 股票问题

###  1.1 [122 . 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)
𝑓 [𝑖,𝑗]代表第 𝑖天结束时, 持有 𝑗股(0或1)
←表示max更新
* 买: 𝑓 [𝑖, 1] ← 𝑓 [𝑖 − 1,0] − 𝑝𝑟𝑖𝑐𝑒𝑠 [𝑖]
* 卖: 𝑓 [𝑖, 0] ← 𝑓 [𝑖 − 1,1] + 𝑝𝑟𝑖𝑐𝑒𝑠[𝑖]
* 不买不卖: 𝑓 [𝑖,𝑗] ← 𝑓 [𝑖 − 1,𝑗]


```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        prices = [0] + prices
        f = [[-1e9, -1e9] for _ in range(len(prices))]
        f[0][0] = 0
        for i in range(1, len(prices)):
            f[i][1] = max(f[i][1], f[i - 1][0] - prices[i])
            f[i][0] = max(f[i][0], f[i - 1][1] + prices[i])
            for j in range(2):
                f[i][j] = max(f[i][j], f[i-1][j])
        return f[-1][0]
```




###  1.2 [188 . 买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)


𝑓 [𝑖,𝑗]代表第 𝑖天结束时, 持有 𝑗股(0或1), k表示交易k次
←表示max更新
* 买: 𝑓 [𝑖, 1,k] ← 𝑓 [𝑖 − 1,0,k-1] − 𝑝𝑟𝑖𝑐𝑒𝑠 [𝑖]
* 卖: 𝑓 [𝑖, 0,k] ← 𝑓 [𝑖 − 1,1,k] + 𝑝𝑟𝑖𝑐𝑒𝑠[𝑖]
* 不买不卖: 𝑓 [𝑖,𝑗,k] ← 𝑓 [𝑖 − 1,𝑗,k]


```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n, c, ans = len(prices), k, 0
        prices = [0] + prices
        f = [[[-1e9]* (c + 1) for _  in range(2)] for _ in range(n + 1)] # c+1考虑到k=0情况
        f[0][0][0] = 0  #初始化

        for i in range(1, n + 1):
            for j in range(2):
                for k in range(c + 1):
                    f[i][j][k] = f[i - 1][j][k] 
                    if k > 0 and j == 1:    #买入需要k>0
                        f[i][1][k] = max(f[i][1][k], f[i - 1][0][k - 1] - prices[i])
                    if j == 0:
                        f[i][0][k] = max(f[i][0][k], f[i - 1][1][k] + prices[i])

        for k in range(c + 1):
            ans = max(ans, f[-1][0][k])
        return ans
```





###  1.3 [714 . 买卖股票的最佳时机含手续费](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/ )

* 思路1: fee只对值有影响 >>>> 在122基础上, 增加买入手续费即可

```python
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        prices = [0] + prices
        f = [[-1e9, -1e9] for _ in range(len(prices))]
        f[0][0] = 0
        for i in range(1, len(prices)):
            f[i][1] = max(f[i][1], f[i - 1][0] - prices[i] - fee)  #手续费
            f[i][0] = max(f[i][0], f[i - 1][1] + prices[i])
            for j in range(2):
                f[i][j] = max(f[i][j], f[i-1][j])
        return f[-1][0]
```



###  1.4 [309 . 最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)


* ==思路==: 加入参数l记录是否进入冷冻期



```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        prices = [0] + prices
        f = [[[-1e9, -1e9]for _ in range(2)] for _ in range(len(prices))]
        f[0][0][0] = 0
        for i in range(1, len(prices)):
            for j in range(2):
                for l in range(2):
                    f[i][1][0] = max(f[i][1][0], f[i - 1][0][0] - prices[i]) # 买之前必非冷冻
                    f[i][0][1] = max(f[i][0][1], f[i - 1][1][0] + prices[i]) # 卖之后必冷冻
                    f[i][j][0] = max(f[i][j][0], f[i-1][j][l])  # 无论之前是否冷冻
        return max(f[-1][0][0], f[-1][0][1])
```



## 2 股票问题优化对比

### 2.1 对比贪心
* 无交易次数限制 >>>> 可贪心 (如122)
往后看一天就知道今天怎么操作，**局部最优**  -->>  **全局最优** 需要证明

* 有交易k次限制 >>>> 不能贪心(如188)
由于局部最优可能导致次数的浪费 >>> 如明天小幅下降后天大幅上涨情况: 贪心-->>**小幅下降卖出再买回**
**往后看到底才有可能知道今天怎么操作，决策是基于全局考量的**


解题路线:<font color=red face="微软雅黑" size = 4>   蛮力搜索---（同类子问题）---> 分治---（最优子结构）---> 动态规划</font>



### 2.2 列表法写状态方程
* 对于股票买卖的状态方程`f[i][j][k][l]`
 之前思路:  **考虑入边**, 即`f[i][j][k][l]`**如何计算**(由之前状态计算)
 另一思路: **考虑出边**, 即`f[i][j][k][l]`可更新哪些状态


![在这里插入图片描述](https://img-blog.csdnimg.cn/60db05de74d64bd8adb61530e368c6ff.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


* 309题列表考虑出边并优化:

 

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        prices = [0] + prices
        f = [[[-1e9, -1e9]for _ in range(2)] for _ in range(len(prices))]
        f[0][0][0] = 0
        for i in range(len(prices) - 1):
            for j in range(2):
                for l in range(2):
                    if f[i][j][l] == -1e9: continue
                    if j == 0 and l == 0: #达到下一天买的条件
                        f[i + 1][1][0] = max(f[i + 1][1][0], f[i][j][l] - prices[i + 1])
                    if j == 1 and l == 0: #持仓可卖出
                        f[i + 1][0][1] = max(f[i + 1][0][1], f[i][j][l] + prices[i + 1]) 
                    f[i + 1][j][0] = max(f[i + 1][j][0], f[i][j][l])  # 无论之前是否冷冻
        return max(f[-1][0][0], f[-1][0][1])
```




### 2.3 空间优化

* 由于以上无论何种条件何种方法 >> 更新只发生在两行之间 : `f[i]` 与 `f[i- 1]`
可利用**滚动数组**优化空间




### 2.4 相关题目
#### 2.4.1 [198 . 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

𝑓 表示计划偷窃前`i`座房屋，第`i`座房屋的闯入情况为`j`（0-未闯入，1-闯入）时的最大收益
不偷 --- 之前可以偷过也可没偷 : 𝑓 [𝑖, 0] = max (𝑓 [𝑖 − 1,1] , 𝑓[𝑖 − 1,0])
偷--- 之前必须没偷 : 𝑓 [𝑖, 1] = 𝑓[ 𝑖 − 1,0 ]+ 𝑛𝑢𝑚𝑠[𝑖]



```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        nums = [0] + nums
        f = [[-1e9, -1e9] for _ in range(n + 1)]
        f[0][0] = 0
        for i in range(1, n + 1):
            for j in range(2):
                f[i][1] = f[i - 1][0] + nums[i]
                f[i][0] = max(f[i - 1][0], f[i - 1][1])
        return max(f[-1][0], f[-1][1])
```



#### 2.4.2 [213 . 打家劫舍2](https://leetcode-cn.com/problems/house-robber-ii/)


* 思路:  1与n号向邻 >>>  原本算法会出现1和n都偷不合法情况 >> 两次DP


```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:return nums[0]
        nums = [0] + nums
        f = [[-1e9, -1e9] for _ in range(n + 1)]
        f[1][0] = 0                  # j=0 不偷1
        for i in range(2, n + 1):
            for j in range(2):
                f[i][1] = f[i - 1][0] + nums[i]
                f[i][0] = max(f[i - 1][0], f[i - 1][1])
        ans_1 = max(f[-1][0], f[-1][1])

        # 再计算不偷n
        f[1][0] = 0
        f[1][1] = nums[1]  #可偷1 f[1][1]合法
        for i in range(2, n + 1):
            for j in range(2):
                f[i][1] = f[i - 1][0] + nums[i]
                f[i][0] = max(f[i - 1][0], f[i - 1][1])
        return max(ans_1, f[-1][0])  # 不偷n
```



#### 2.4.3 [72 . 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

* 思路:  只考虑次数最小 >>> 对插, 删, 换 三个操作取min即可


```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        word1 = " " + word1
        word2 = " " + word2
        f = [[-1e9] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):  # i 到 0 个字符删i次(赋值是因为需要使用)
            f[i][0] = i
        for j in range(m + 1):
            f[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                f[i][j] = min(f[i][j - 1] + 1, 
                              f[i - 1][j] + 1, 
                              f[i - 1][j - 1] + (word1[i] != word2[j]))  # 分别代表插删换操作
        return f[-1][-1]
```



## 3 背包问题
1. 0/1 背包
给定`N`个物品，其中第`i`个物品的体积为$V_i$,价值为$W_i$ 
有一容积为`M`的背包，要求选择一些物品放入背包，使得物品总体积不超过M的前提下，物品的 价值总和最大
$F[i, j]$表示从前i个物品中选了体积为j所得物品最大值
$$
F[i, j]=\max \left\{\begin{array}{cc}
F[i-1, j] & \text { 不选第 } i \text { 个物品 } \\
F\left[i-1, j-V_{i}\right]+W_{i} & \text { if } j \geq V_{i} \quad \text { 选第 } i \text { 个物品 }
\end{array}\right.
$$



* 例题 [416 . 分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)
  * 思路:  `f[i][j] `表示在第 i 个数为止选出一些数求和,  达到 j 是否可行(bool)
  `f[i][j]  = f[i-1][j - nums[i]]   or f[i - 1][j]`

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n, sum_num = len(nums), 0
        nums = [0] + nums
        for _ in range(n + 1):
            sum_num += nums[_]
        if sum_num % 2 != 0: return False
        f = [False] * (sum_num // 2 + 1)
        f[0] = True

        for i in range(1, n + 1):
            for j in range(sum_num // 2, nums[i] - 1, -1):  #考虑到f[j]需要从上一个更新, 倒过来求
                f[j] = f[j - nums[i]] or f[j]
        return f[sum_num // 2]

```



2. 完全背包

给定N种物品7其中第`i`种物品的体积为$V_i$,价值为$W_i$ ，并且有无数个 
有一容积为M的背包，要求选择若干个物品放入背包，使得物品总体积不超过M的前提下，物品 的价值总和最大

$F[i, j]$表示从前i个物品中选了体积为j所得物品最大值


$$
F[i, j]=\max \left\{\begin{array}{c}   F[i-1, j] \\
F\left[i, j-V_{i}\right]+W_{i} \quad \text { if } j \geq V_{i} \quad \text { 从第 } i \text { 种物品中选一个 }
\end{array}\right.
$$


[518 . 零钱兑换 II](https://leetcode-cn.com/problems/coin-change-2/)


```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        coins = [0] + coins
        f = [0] * (amount + 1)
        f[0] = 1

        for i in range(1, n + 1):
            for j in range(coins[i], amount + 1): # j从大于coin[i]开始 
                f[j] += f[j - coins[i]]
        return f[amount]
```


## 4 作业
### 4.1 [279 . 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)


* ==思路==:  看成完全背包问题,  组成完全平方数看成物品,  个数最少看成目标, 和等于n看成前提

`f[i, j]`表示前` i` 个完全平方数 选出和为` j `, 最少的个数
`f[i, j] =  min (f[i, j - nums[i]], f[i, j])`


```python
class Solution:
    def numSquares(self, n: int) -> int:
        nums = [i*i for i in range(1, int(n**(1/2)) + 1)]
        f = [1e4] * (n + 1)
        f[0] = 0

        for i in range(len(nums)):
            for j in range(nums[i], n + 1):  # 从nums[i]开始
                f[j] = min(f[j - nums[i]] + 1, f[j])
        return f[-1]
```




### 4.2   [55 . 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)


* ==思路==:  倒着使用dp 利用f[i]来记录数组中可达情况

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        f = [False] * n
        f[-1] = True

        j = n - 1
        for i in range(n - 2, -1, -1):
            if j - i <= nums[i]:     #符合条件说明i 可以到达j 
                f[i] = True
                j = i      #更新j  计算之前元素能否到j
        return f[0]  # 第一个元素为True 说明可由第一到最后
```


### 4.3   [45 . 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

* ==思路==:   正向dp, 考虑出度>>> i 能到达哪些点

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        f = [1e4] * n
        f[0] = 0

        for i in range(n - 1):
            for j in range(i + 1, nums[i] + i + 1):
                if j < n and j - i <= nums[i]:
                    f[j] = min(f[i] + 1, f[j])
                
        return f[-1]
```

