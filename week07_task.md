# 七、贪心动规

![在这里插入图片描述](https://img-blog.csdnimg.cn/205c74c5f89f497387f8957ae461effc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----


@[TOC](目录)



---
#### <font color=red face="微软雅黑">来源</font>
  [极客时间2021算法训练营](https://u.geekbang.org/lesson/194?article=419794&utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=u_nav_web)

作者:  李煜东


----
## 1 贪心

###  1.1 基本思想
> 贪心算法(GreedyAlgorithm)是一种 
(1)在每一步选柽中都采取在当前状态下的最优决策(局部最优) 
(2)并希望由此导致的最终结果也是全局最优
的算法


* **贪心和动规不同**:  不对整个状态空间进行 遍历或计算，而是始终按照局部最优选择执行下去，不再回头。

* **状态空间角度**: 贪心算法实际上是在状态空间中按局部最优策略找了一条路径。


--- 
* 使用贪心法要求问题的<font color=red face="微软雅黑">整体最优性可以由局部最优性导出</font>。贪心算法的正确性需要证明, 常见的证明手段有:

  * 微扰 (邻项交换)
    证明在任意局面下, 任何对局部最优策略的微小改变都会造成整体结果变差。经常用于以 “排序” 为贪心策略的证明。
  * 范围缩放(拓展范围)
    证明任何对局部最优策略作用范围的扩展都不会造成整体结果变差。
  * **决策包容性**
    证明在任意局面下, 作出局部最优决策以后, 在问题状态空间中的可达集合包含了作出其他任何决策后的可达集合。换言之, 这个**局部最优策略提供的可能性包含其他所有策略提供的可能性**。
  * 反证法
  * 数学归纳法


---

* 例子
零钱兑换：贪心 
根据我们平时找钱的思路，一般我们会先考虑面值大的，零钱再用面值小的凑齐 "每次都选尽量大的面值" 就是一个贪心思想


![在这里插入图片描述](https://img-blog.csdnimg.cn/1e7949d14b1b481694e3851ba215f644.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


---

### 1.2 相关例题
#### 1.2.1 [860 . 柠檬水找零](https://leetcode-cn.com/problems/lemonade-change/)


* ==思路==:  动态维护币值数量 >>>>  贪心: 先找面值大的

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        dic = defaultdict(int)
        for bill in bills:
            dic[bill] += 1
            curr = bill - 5
            for money in [10, 5]:  #从找10元开始考虑
                while curr >= money and dic[money] > 0:
                    dic[money] -= 1
                    curr -= money
            if curr != 0:
                return False
        return True
```

#### 1.2.2 [455 . 分发饼干](https://leetcode-cn.com/problems/assign-cookies/)

* ==思路==: 贪心 >>> 大饼先满足需求大的孩子g[i],  剩下更易满足


**决策包容性证明**:贪心局部最佳>>全局最佳：一块饼干总是想要满足一个孩子的，满足胃口更大的孩子，未来的可能性包含了满足 胃口更小孩子的可能性 


```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        count,ans = 0, 0
        for child in g:
            while count < len(s) and s[count] < child:
                count += 1
            if count < len(s):
                ans += 1
                count += 1
        return ans
```



#### 1.2.3 [122 . 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

* ==思路==: 获得所有`prices[i] - prices[i - 1] > 0`区间的收益
**决策范围扩展** 
在思考贪心算法时，有时候不容易直接证明局部最优决策的正确性 **此时可以往后多扩展一步，有助于对当前的决策进行验证**
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        for i in range(1, len(prices)):
            ans += max(prices[i] - prices[i-1], 0)
        return ans
```



#### 1.2.4 [45 . 跳跃游戏 II](https://leetcode-cn.com/problems/jump-game-ii/)

* ==思路==: 查看后两步的可能性
**决策包容性**：同样都是跳1步，从a跳到"能跳得更远"的b,未来的可达集合包含了跳到其他 b的可达集合，所以这个局部最优决策是正确的。


```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        ans, now = 0, 0
        while now < len(nums) - 1:
            right = now + nums[now]  # right是now位置能达到最大位置
            if right >= len(nums) - 1:
                return ans +1
            nextNow = now
            nextRight = right
            for i in range(now + 1, right + 1): # now 可以达到范围[now + 1, right]
                if i + nums[i] > nextRight:        #使得 now下下个可达位置最远
                    nextNow = i
                    nextRight = i + nums[i]
            now = nextNow
            ans += 1
        return ans
```



#### 1.2.5 [1665 . 完成所有任务的最少初始能量](https://leetcode-cn.com/problems/minimum-initial-energy-to-finish-tasks/)
**邻项交换** 
经常用于以某种顺序"排序"为贪心策略的证明 
证明在任意局面下，任何局部的逆序改变都会造成整体结果变差


* ==思路==:  需要求顺序 >>> 动规 二分需排序  >>> 贪心 >>> 门槛(`task[1]`)高 具有较优先趋势, 能耗(`task[0]`)低同样具有较优先趋势 >>>  考虑`task[0] - task[1]`升序排序

证明:
设`task[0]`为`actual`, `task[1]`为`minimum`;
设做完第`i+2`到`n`个任务所需的初始能量最少为`S` ;
对于两个相邻任务：设第`i`个和第`i+l`个完成的任务分别是`p`和`q`:


![在这里插入图片描述](https://img-blog.csdnimg.cn/6867c6124df54c9ba42720e70f2f5247.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)
考虑$p, q$中先做者所需要的最低能量$S_p, S_q$: 
先做p, 所需要能量:
$$S_p =max(max(minimum[q], S+actual[q])+actual[p], minimum[p])$$
$$  =max(minimum[q] + actual[p], S+actual[q] + actual[p], minimum[p])$$
先做q, 所需要能量:
$$S_q =max(max(minimum[p], S+actual[p])+actual[q], minimum[q])$$
$$=max(minimum[p] + actual[q], S+actual[p] + actual[q], minimum[q])$$

P优先则 >>> 满足 $S_p<S_q$ 即:
$$max(minimum[q] + actual[p], minimum[p]) < max(minimum[p] + actual[q], minimum[q])$$

因为必定有 $minimum[q] + actual[p] > minimum[q]$ 
所以上式等价于 $minimum[q] + actual[p] < minimum[p] + actual[q]$
即 $actual[p] - minimum[p] < actual[q] - minimum[q]$


于是有: >>>**贪心策略：按照`actual - minimum`升序排序，以此顺序完成任务**

```python
class Solution:
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        ans = 0
        tasks.sort(key = lambda x: x[0] - x[1])
        for task in tasks[::-1]:   #倒序考虑完成所需要能量
            ans = max(task[1], ans + task[0])
        return ans
```




## 2 线性动规
>动态规划（英语：Dynamic programming，简称 DP），是一种在数学、管理科学、计算机科学、经济学和生物信息学中使用的，**通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法**。动态规划常常适用于有重叠子问题和最优子结构性质的问题。
* 即 >>> **搜索的优化**
---
* 对于零钱兑换问题(`amount = 18, coins = [10, 9, 1]`)的状态空间:
   * opt(n) = min(opt(n - 1), opt(n - 9),opt(n - 10)) + 1
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/c8cc59eef0cb4f74ad76b1b301b4b627.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

  * 式子本质: **状态+最优化目标+最优化目标之间具有递推关系=最优子结构**



> 动态规划(DP, dynamic programming)是一种对问题的状态空间进行分阶段、有顺序、不重复、 决策性遍历的算法

* 动态规划的关键与前提： 
**重叠子问题**——与递归、分治等一样，要具有同类子问题，用若干维状态表示 
**最优子结构** ——状态对应着一个最优化目标，并且最优化目标之间具有推导关系 
**无后效性**——问题的状态空间是一张有向无环图(可按一定的顺序遍历求解) 
* 动态规划一般采用递推的方式实现 
也可以写成递归或搜索的形式，因为每个状态只遍历一次，也被称为**记忆化搜索**


* **动态规划三要素：阶段、状态、决策**


![在这里插入图片描述](https://img-blog.csdnimg.cn/61628e54ffcd454dbc8b713baed51509.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)



### 2.1 相关题目

#### 2.1.1 [63 . 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)


* 思路:   dfs在此计数问题由于方案数 * 方案数 >>> 时间复杂度较大 >>> 递归,分治思想 >>> 记忆化搜索 >>> 减小时间复杂度

Bottom-up
`f[i,j]`表示从`(i,j)`到`End`的路径数, 如果`(i,j)`是空地,`f[i,j]` = `f[i + l,j] + f[i,j + 1]`
否则`f[i,j] = 0`
![在这里插入图片描述](https://img-blog.csdnimg.cn/56f9d7f31ebd4ab590e1a19efa4f0a13.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

反过来类似的:

Top-down 
`f[i,j]`表示从Start到`(i，j)`的路径数, 如果`(i,j)`是空地,`f[i,j]` = `f[i - 1,j] + f[i,j - 1]`

否则`f[i,j] = 0`

![在这里插入图片描述](https://img-blog.csdnimg.cn/9a9333d5b7084933806177d619d8184e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)




```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        f = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    f[i][j] = 0
                elif i == 0 and j == 0:
                    f[i][j] = 1
                elif i == 0:
                    f[i][j] = f[i][j - 1] 
                elif j == 0:
                    f[i][j] = f[i - 1][j]
                else:
                    f[i][j] = f[i - 1][j] + f[i][j - 1]
        return f[-1][-1]
```



#### 2.1.2  [1143 . 最长公共子序列](https://leetcode-cn.com/problems/longest-common-subsequence/)

* ==思路==:  只关心数量>>> 动规 >>> 观察蛮力搜索状态 >> 寻找变化信息 >>> 确定最优子结构, 边界

`f[i][j]`表示`text1`的前`i`个字符和`text2`的前`j`个字符能组成的LCS的长度
如果 `text1[i] = text2[j]`:   `f[i][j] =  f[i - 1][j-1] + 1`
如果 `text1[i] != text2[j]`:   `f[i][j] =  max(f[i - 1][j], f[i][j-1]  )`


![在这里插入图片描述](https://img-blog.csdnimg.cn/fe333d5df88542859d9504b158f55654.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)


```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        f = [[0]*(n + 1) for _ in range(m + 1)]
        text1 = " " + text1   #防止i-1, j-1越界
        text2 = " " + text2
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i] == text2[j]:
                    f[i][j] = f[i - 1][j - 1] + 1
                else:
                    f[i][j] = max(f[i - 1][j], f[i][j - 1])
        return f[-1][-1]
```


#### 2.1.3 [300 . 最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

* ==思路:==  指数级别状态空间(子集 >> 选或不选)  >>> 在每一状态只关心**结尾数值**  以及 **选择个数**


`f[i]`表示前`i`个数构成的**以`a[i]`为结尾的最长上升子序列的长度**

$$
f[i]=\max _{j<i, a[j]<a[i]}\{f[j]+1\}
$$

边界：`f[i] = 1 (0 <= i < n) `
目标：$\max _{0 \leq i<n}\{f[i]\}$

![在这里插入图片描述](https://img-blog.csdnimg.cn/e650bfa79a0e4b2e9c6f1d0a3d5cbfa4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_12,color_FFFFFF,t_70,g_se,x_16)



```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n, ans = len(nums), 0
        f = [1 for _ in range(n)]
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    f[i] = max(f[j] + 1, f[i])
        for k in range(n):
            ans = max(ans, f[k])
        return ans
```



* 假如记下路径

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n, ans,curr,end = len(nums), [], 0, -1
        f = [1 for _ in range(n)]
        pre = [-1 for _ in range(n)]
        def p(i):
            if pre[i] != -1:
                p(pre[i])
            ans.append(nums[i])
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j] and f[i] < f[j] + 1 :
                    f[i] = f[j] + 1
                    pre[i] = j
        for k in range(n):
            if f[k] > curr:
                end = k
        p(end)
        return ans
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/ac3457bbad804b029f99aa502bb13afa.png)

#### 2.1.4 [53 . 最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)
𝑓[𝑖] 表示以 𝑖 为结尾的最大子序和
`𝑓 [𝑖] = max (𝑓 [𝑖 − 1] + 𝑛𝑢𝑚𝑠 [𝑖] , 𝑛𝑢𝑚𝑠[𝑖])`


```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        ans = nums[0]
        f = [nums[0] for _ in range(n)]
        for i in range(1, n):
            f[i] = max(nums[i], f[i-1] + nums[i])
        for i in range(1, n):
            ans = max(ans, f[i])
        return ans
```


#### 2.1.5 [152 . 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)



 * ==思路==:
**考虑到负负得正情况, 不能只考虑当前乘积最大作为最优**:
max和min一起作为代表，才满足最优子结构！
𝑓𝑚𝑎𝑥 [𝑖 ] , 𝑓𝑚𝑖𝑛[𝑖] 表示以 𝑖 为结尾的乘积最大、最小子数组
𝑓𝑚𝑎𝑥[ 𝑖 ]= max (𝑓𝑚𝑎𝑥 [𝑖 − 1 ]∗ 𝑛𝑢𝑚𝑠 [𝑖] , 𝑓𝑚𝑖𝑛 [𝑖 − 1] ∗ 𝑛𝑢𝑚𝑠 [𝑖] , 𝑛𝑢𝑚𝑠[𝑖])
𝑓𝑚𝑖𝑛[ 𝑖 ]= min (𝑓𝑚𝑎𝑥 [𝑖 − 1] ∗ 𝑛𝑢𝑚𝑠 [𝑖] , 𝑓𝑚𝑖𝑛 [𝑖 − 1] ∗ 𝑛𝑢𝑚𝑠 [𝑖] , 𝑛𝑢𝑚𝑠[𝑖])


```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        ans = nums[0]
        fmin = [nums[0] for _ in range(n)]
        fmax = [nums[0] for _ in range(n)]
        for i in range(1, n):
            fmax[i] = max(nums[i], fmax[i-1] * nums[i], fmin[i-1] * nums[i])
            fmin[i] = min(nums[i], fmax[i-1] * nums[i], fmin[i-1] * nums[i])
        for i in range(1, n):
            ans = max(ans, fmax[i])
        return ans
```



## 3 作业
### 3.1 [70 . 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)
* 思路: n阶上一位n-1或n-2 :  f(n) = f(n-1) + f(n-2)
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        f = [0 for _ in range(n + 1)]
        f[0] = f[1] = 1
        for i in range(2, n + 1):
            f[i] = f[i - 1] + f[i - 2]
        return f[-1]
```

### 3.2 [120 . 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

* ==思路==: 从下至上构建 f[i - 1][j] = triangle[i - 1][j] + min(f[i][j], f[i][j + 1])


```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        f = [[0]* (i + 1) for i in range(len(triangle))]
        f[-1] = triangle[-1]
        for i in range(len(triangle) - 1, 0, -1):
            for j in range(len(triangle[i])-1):
                f[i-1][j] = triangle[i - 1][j] + min(f[i][j], f[i][j + 1])
        return f[0][0]
```



###  3.3 [673 . 最长递增子序列的个数](https://leetcode-cn.com/problems/number-of-longest-increasing-subsequence/)


* ==思路==:  和300题类似, 不过记得需要绩效个数


```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        if len(nums) == 1:return 1
        n = len(nums)
        max_len, ans = 0, 0
        f = [1 for _ in range(n)]   # 递增序列长度
        count = [1 for _ in range(n)]  # 递增序列的个数
        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j] and f[i] < f[j] + 1: #若当前长度大于之前记录长度 更新长度
                    f[i] = f[j] + 1
                    count[i] = count[j]
                elif nums[i] > nums[j] and f[i] == f[j] + 1: #若历史长度与某一当前j长度一致 更新count
                    count[i] += count[j]
            max_len = max(max_len, f[i])
        for i in range(n):
            if f[i] == max_len:
                ans += count[i]
        return ans
```

## 参考资料
1. <https://cloud.tencent.com/developer/article/1817113>看一遍就理解：动态规划详解
2. <>
