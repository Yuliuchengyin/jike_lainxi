# 九、动规三字典树

![在这里插入图片描述](https://img-blog.csdnimg.cn/205c74c5f89f497387f8957ae461effc.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

----


@[TOC](目录)



---
#### <font color=red face="微软雅黑">来源</font>
  [极客时间2021算法训练营](https://u.geekbang.org/lesson/194?article=419794&utm_source=u_nav_web&utm_medium=u_nav_web&utm_term=u_nav_web)

作者:  李煜东


----
## 1 动规优化
* 在动态规划中有 **状态变量**和**决策变量** 
  * **分离状态变量和决策变量**。当循环多于两重时，关注最里边的两重循环，把外层看作定值。
  *  对于一个状态变量, **决策变量的取值范围称为"决策候选集合"**,观察这个集合随着状态变量 
的变化情况 
> 一旦发现冗余，或者有更高效维护"候选集合"的数据结构，就可以省去一层循环扫描!



### 1.1 [1499 . 满足不等式的最大值](https://leetcode-cn.com/problems/max-value-of-equation/)


* ==思路==:  固定`i` , 观察 `j`(决策)的变化 >> O(n^2)变为O(n)
1.   $y_i + y_j + |x_i - x_j|$ **可知其和值与 $i, j$ 顺序无关** , 假**设 $j < i$ , 则另一个限制条件是 $x_i - x_j < k$**;   若 $i$增大,  $j$的取值范围也增大 ,  因为 $j <= i +1$(上界增大), $x_j > x_i - k$(下界随着x_i增大必须增大)  >>>>>  维护 `max(y[j] - x[j])`  >>> 滑动窗口
2. 利用单调队列q 存储q[0]存储 `j` 最优值, 使得 `y[j] - x[j]`最大 >>>  即 使得 `y[j] - x[j]`单调递减
3. 实现 : 
第一步 : 通过 `x[j] >= x[i] - k`   >>> 筛选删除过期 `j`;
第二步: 通过`q[0]` (最佳决策`j`)更新 `ans =  max(ans, x[i] + y [i] + y[j] - x[j])`
第三步: 维护单调队列 >>> 使得`y[j] - x[j]`递减


```python
class Solution:
    def findMaxValueOfEquation(self, points: List[List[int]], k: int) -> int:
        from collections import deque
        q = deque()
        ans = -1e9
        # j上界: j <= i 
        # j下界: x[j] >= x[i] - k
        for i in range(len(points)):
            # 当 j 越下界 删除过期的j
            while (q and points[q[0]][0] < points[i][0] - k):
                q.popleft()
            # 通过合法的j (q[0]) 更新最佳答案 x[i] + y[i] + y[j] - x[j]
            if q:
                ans = max(ans, points[i][0] + points[i][1] 
                               + points[q[0]][1] - points[q[0]][0])
            # 维护q单调队列 y[j] - x [j]递减
            while (q and points[q[-1]][1] - points[q[-1]][0] <= points[i][1] - points[i][0]):
                q.pop()
            q.append(i)
        return ans
```



### 1.2 [918 . 环形子数组的最大和](https://leetcode-cn.com/problems/maximum-sum-circular-subarray/)

* ==思路==:  假设无环线性区间 >> 设 `F[i] = S[i] - S[j]`,其中 S为前缀和 >> 即找到最小的`S[j]` , `i`看做状态,`j`看做决策; 
$$
F[i]=S[i]-\min _{i-n \leq j<i}\{S[j]\}
$$
对于环形 >>> 可复制一倍变长为2n数组, 再以**n为滑动窗口**取前缀和之差 >>>  最优F



```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        from collections import deque
        q = deque()

        n = len(nums)
        nums = [0] + nums
        ss = [0]*(2 * n + 1)
        ans = -1e9
        # 先求出长2n数组的前缀和ss
        for i in range(1, n + 1):
            ss[i] = ss[i - 1] + nums[i]
        for i in range(n + 1, 2 * n + 1):
            ss[i] = ss[i - 1] + nums[i - n]

        #单调队列q, q[0]为最佳决策j, 使得ss[j]最小, ss[i] - ss[j]最大
        for i in range(1, 2 * n + 1):
            # 1. 根据范围 i - n <= j <= i - 1 , 判断j(q[0])合法性
            while q and q[0] < i - n:
                q.popleft()
            # 决策合法, 更新ans
            if q:
                ans = max(ans, ss[i] - ss[q[0]])
            # 维护单调队列q, 保证ss[j]递增
            while q and ss[q[-1]] >= ss[i]:
                q.pop()
            q.append(i)
        return ans
```



* ==思路二==:  环形最大子段和 >>> 转换为 求 `total_sum - 线性最小子段和`


```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        nums = [0] + nums
        s = [0] * (n + 1)
        # 1.先求前缀和
        for i in range(1, n + 1):
            s[i] = s[i - 1] + nums[i]

        # 2. 先求线性最大子段和 ans
        temp = 1e9
        ans = -1e9
        for i in range(1, n + 1):  # i从1开始代表可以取到 S[n] - S[0]
            temp = min(temp, s[i - 1])
            ans = max(ans, s[i] - temp)
        # 3. 求线性最小子段和 ansMin
        temp = -1e9
        ansMin = 1e9
        for i in range(2, n):     # s[1:n]  为了防止全取
            temp = max(temp, s[i - 1])
            ansMin = min (ansMin, s[i] - temp)
        # 比较 ans 和 S[n] - ansMin
        return max(ans, s[n] - ansMin)
```



## 2 区间动态优化

### 2.1 [312 . 戳气球](https://leetcode-cn.com/problems/burst-balloons/)
* ==思路==:  
  1.若考虑先戳哪一个气球`q`,   >>>   如戳破 `q` 后 分为`[1: q]`和`[q+1: n]`相邻气球发生改变 >> 和原区间相比并非同类子问题
  2.因此考虑最后戳气球`q` >> `[1: q]`和`[q+1: n]`子区间和原区间`[l,r]`同样
![在这里插入图片描述](https://img-blog.csdnimg.cn/044f625e327e40758734f6a0b792cfaf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5L2Z5p-z5oiQ6I2r,size_20,color_FFFFFF,t_70,g_se,x_16)

* `f[l,r]`表示戳破闭区间l〜r之间的所有气球，所获硬币的最大数量

$$
f[l, r]=\max _{l \leq p \leq r}\{f[l, p-1]+f[p+1, r]+n u m s[p] * n u m s[l-1] * n u m s[r+1]\}
$$



```python
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        n = len(nums)
        nums = [1] + nums + [1]
        f = [[0]*(n + 2) for _ in range(n + 2)]
        for size in range(1, n + 1):
            for l in range(1, n - size + 2): # l边界最多取到 n - size + 1
                r = l + size - 1  # l + r = n
                for p in range(l, r + 1):
                    f[l][r] = max(f[l][r], f[l][p - 1] + f[p + 1][r] +
                                           nums[p]*nums[l - 1]*nums[r + 1])
        return f[1][n]    
```
## 3 树形动态规划

>树形动态规划与线性动态规划没有本质区别 
其实只是套在深度优先遍历里的动态规划（在DFS的过程中实现DP


[337 . 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

* ==思路==: 用动规考虑每一个子树的情况
`f[x, 0]`表示以`x`为根的子树，在不打劫`x`的情况下，能够盗取的最高金额 
`f[x, 1]`表示以`x`为根的子树，在打劫`x`的情况下，能够盗取的最高金额


$$
\begin{gathered}
f[x, 0]=\sum_{y \text { is a son of } x} \max (f[y, 0], f[y, 1]) \\
f[x, 1]=\operatorname{val}(x)+\sum_{y \text { is a son of } x} f[y, 0]
\end{gathered}
$$

```python
class Solution:
    def rob(self, root: TreeNode) -> int:
        if not root: return 0
        def dfs(root):
            if not root:return [0, 0]
            l = dfs(root.left)
            r = dfs(root.right)
            f = [0, 0]
            # f[0], f[1]分别代表偷根和不偷根
            f[0] = max(l[0], l[1]) + max(r[0], r[1])
            f[1] = l[0] + r[0] + root.val
            return f
        return max(dfs(root))
```


## 4 字典树
> Trie树，即字典树，又称单词查找树或键树，是一种树形结构，是一种哈希树的变种。典型应用是用于统计和排序大量的字符串（但不仅限于字符串），所以经常被搜索引擎系统用于文本词频统计。


**优点**：利用字符串的公共前缀来减少查询时间，**最大限度地减少无谓的字符串比较**。

* ==性质==

1. 结点本身不保存完整单词。
2. 从根结点到某一结点，路径上经过的字符连
接起来，为该结点对应的单词。
3. 每个结点出发的所有边代表的字符都不相同。
4. 结点用于存储单词的额外信息（例如频次）。


* ==实现方式==
   * 字符集数组法（简单）
每个结点保存一个长度 定为字符集大小（例如26）的数组，以字符为下标，保存指向的结点
空间复杂度为`O（结点数*字符集大小）`，查询的时间复杂度为`O（单词长度）`
**适用于较小字符集，或者单词短、分布稠密的字典**
  * 把每个结点上的字符集数组改为一个映射（词频统计：hash map,排序：ordered map） 
空间复杂度为`O（文本字符总数）`，查询的时间复杂度为`O（单词长度）`，但常数稍大一些 
适用性更广		 
 


1. [208 . 实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)


* ==思路==:  利用字典嵌套 + insert 计数实现
```python
class Trie:

    def __init__(self):
        self.root = [0, {}]

    def insert(self, word: str) -> None:
        self.find(word, True, True)

    def search(self, word: str) -> bool:
        return self.find(word, True, False)

    def startsWith(self, prefix: str) -> bool:
        return self.find(prefix, False, False)

    def find(self, s, match, insert):
        curr = self.root
        for ch in s:
            if ch not in curr[1]:
                if not insert:
                    return False
                curr[1][ch] = [0, {}]
            curr = curr[1][ch]
        if insert:
            curr[0] += 1
        return curr[0] > 0 if match else True 

```




## 5 作业
1.[516 . 最长回文子序列](https://leetcode-cn.com/problems/longest-palindromic-subsequence/)
* ==思路:==
 dp[i][j]：字符串s在[i, j]范围内最长的回文子序列的长度为dp[i][j]。
 `s[i] = s[j]`时   >>
`dp[i][j] = dp[i + 1][j - 1] + 2`;
 `s[i] != s[j]` 时  >>
 `dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])`

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0]*n for _ in range(n)]
        for i in range(n-1, -1, -1):
            dp[i][i] = 1
            for j in range(i+1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j-1])
        return dp[0][-1]
```



2. [124 . 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

* ==思路==:  树形动规  +  dfs

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.res = -1e9
        def dfs(root):
            if not root:return 0
            l = dfs(root.left)
            r = dfs(root.right)

            cur = root.val + max(0, l) + max(0, r) 
            self.res = max(self.res, cur)                       #更新res
            return root.val + max(max(0,l), max(0,r))
        dfs(root)
        return self.res
```

