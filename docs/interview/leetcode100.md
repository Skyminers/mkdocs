---
date: 2024-08-26
tags:
    - C++
---

# Leetcode 100 热题题解与代码

一句话题解，简要概括题目思路解法，也会提供对应代码。

---

## 1. [两数之和](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked)

一个数字确定时另一个数字的值也就确定了，可以用 hashmap 判断另一个数字是否存在，时间复杂度 O(n)。C++ 中可以用 `unordered_map` 实现。

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> mp;
        for (int i  = 0; i < nums.size(); ++ i) {
            auto it = mp.find(target - nums[i]);
            if (it != mp.end()) {
                return {it->second, i};
            }
            mp[nums[i]] = i;
        }
        return {};
    }
};
```

---

## 2. [两数相加](https://leetcode.cn/problems/add-two-numbers/description/?envType=study-plan-v2&envId=top-100-liked)

模拟，为了方便，创建了一个不含数值的头节点来避免判断。

```cpp
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        int x = 0;
        ListNode head = ListNode();
        ListNode *num = &head;
        while(l1 != nullptr || l2 != nullptr || x) {
            num->next = new ListNode();
            num = num->next;

            num->val = x;
            if (l1 != nullptr) {
                num->val += l1->val;
                l1 = l1->next;
            }
            if (l2 != nullptr) {
                num->val += l2->val;
                l2 = l2->next;
            }

            if (num->val >= 10) {
                x = 1;
                num->val -= 10;
            } else x = 0;
        }
        num = head.next;
        return num;
    }
};
```

---

## 3. [无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=top-100-liked)

滑动窗口，用一个 HashSet 记录已经出现的字符，每次右端点向右移动时判断是否重复，如果重复则左端点向右移动。

```cpp
class Solution {
public:
    unordered_set<int> hash;
    int lengthOfLongestSubstring(string s) {
        hash.clear();
        int ans = 0;
        for (int r = 0, l = 0; r < s.size(); ++ r) {
            while(hash.contains(s[r])) {
                hash.erase(s[l++]);
            }
            hash.insert(s[r]);
            ans = max(ans, (int)hash.size());
        }
        return ans;
    }
};
```

---

## 4. [寻找两个正序数组的中位数](https://leetcode.cn/problems/median-of-two-sorted-arrays/description/?envType=study-plan-v2&envId=top-100-liked)

如果可以同时在两个数组中应用合理的划分，将两个数组各分为两部分再合并起来。合并起来的数组中，左半部分数字个数为 $\frac{(n + m + 1)}{2}$ 时，中位数是左半部分的最大值和右半部分的最小值的平均值（如果是奇数的话就直接是左半部分的最大值）。可以通过二分的方式枚举第一个数组的分割点，这样的话第二个数组的分割点就可以直接计算确定。

```cpp
#define inf 0x3fffffff
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        if (nums1.size() > nums2.size()) swap(nums1, nums2);
        int n = nums1.size(), m = nums2.size();
        int l = 0, r = n;
        int split1 = 0, split2 = 0;
        int lmax = -1, rmin = -1;
        while (l <= r) {
            split1 = l+r >> 1;
            split2 = (n + m + 1) / 2 - split1;
            int a = split1-1 < 0 ? -inf : nums1[split1-1];
            int b = split1 >= n ? inf : nums1[split1];
            int x = split2-1 < 0 ? -inf : nums2[split2-1];
            int y = split2 >= m ? inf : nums2[split2];

            if (a <= y && x <= b) {
                lmax = max(a, x);
                rmin = min(b, y);
                break;
            }
            if (a > y) {
                r = split1 - 1;
            } else {
                l = split1 + 1;
            }
        }
        if ((n + m) % 2 == 0) return (double)(lmax + rmin) / 2;
        else return lmax;
    }
};
```

---

## 5. [最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/description/?envType=study-plan-v2&envId=top-100-liked)

manacher 算法

```cpp
class Solution {
    string prework(string s) {
        string ret = "";
        for (int i = 0; i < s.size(); ++ i) {
            ret += '#';
            ret += s[i];
        }
        ret += "#";
        return ret;
    }
    vector<int>p;
public:
    string longestPalindrome(string s) {
        string manaStr = prework(s);
        p.clear();
        p.resize(manaStr.size());
        int R = -1, C = -1;
        int ans = 0;
        for (int i = 0; i < manaStr.size() - 1; ++ i) {
            p[i] = R > i ? min(R - i ,p[2*C-i]) : 1;
            while(i + p[i] < manaStr.size() && i - p[i] >= 0) {
                if (manaStr[i+p[i]] == manaStr[i-p[i]]) ++ p[i];
                else break;
            }
            if (i+p[i] > R) {
                R = i+p[i];
                C = i;
            }
            ans = max(ans, p[i]-1);
        }
        for (int i = 0;i < manaStr.size(); ++ i) {
            if (ans == p[i]-1) {
                int idx = i / 2 - ans / 2;
                return s.substr(idx, ans);
            }
        }
        printf("ans = %d\n" ,ans);
        return "";
    }
};
```

---

## 11. [盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/?envType=study-plan-v2&envId=top-100-liked)

贪心，从两侧用双指针向中间靠拢，每次移动更低的一边，这样才更可能向更好的方向发展。

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0, j = height.size()-1;
        int ans = 0;
        while (i < j) {
            ans = max(ans, min(height[i], height[j]) * (j-i));
            if (height[i] < height[j]) ++ i;
            else --j;
        }
        return ans;
    }
};
```

---

## 15. [三数之和](https://leetcode.cn/problems/3sum/description/?envType=study-plan-v2&envId=top-100-liked)

首先排序，然后枚举三元组的第一个数字，再枚举第二个数字，第三个数字会在第二个数字从前往后枚举的过程中逐渐从后往前移动，维护指针。

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());

        vector<vector<int>> ans;
        ans.clear();

        for (int i = 0;i < nums.size(); ++ i) {
            if (i != 0 && nums[i] == nums[i-1]) continue;

            int k = nums.size()-1;

            for (int j = i+1; j < nums.size(); ++ j) {

                if (j > i+1 && nums[j] == nums[j-1]) continue;
                while(k > j && nums[i] + nums[j] + nums[k] > 0) -- k;
                if (k <= j) break;

                if (nums[i] + nums[j] + nums[k] == 0) {
                    vector<int> tmp; tmp.clear();
                    tmp.push_back(nums[i]);
                    tmp.push_back(nums[j]);
                    tmp.push_back(nums[k]);
                    ans.push_back(tmp);
                }
            }
        }
        return ans;
    }
};
```

---

## 17. [电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/?envType=study-plan-v2&envId=top-100-liked)

搜索枚举

```cpp
class Solution {
    vector<string> ans;
    string res;
public:
    void dfs(string &str, int idx) {
        static string mapping[] = {"", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        if (idx == str.size()) {
            ans.push_back(res);
            return ;
        }
        string mapString = mapping[str[idx] - '1'];
        for (int i = 0; i < mapString.size(); ++ i) {
            res += mapString[i];
            dfs(str, idx+1);
            res.pop_back();
        }

    }
    vector<string> letterCombinations(string digits) {
        ans.clear();
        if (digits.empty()) return ans;
        res = "";
        dfs(digits, 0);
        return ans;
    }
};
```

---

## 19. [删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?envType=study-plan-v2&envId=top-100-liked)

用一个数组把整个链表存了下来，说实话可以压缩到只存最后 n+1 个结点，但是这就需要再维护一个链表，有点懒。

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        vector<ListNode*> listnode;
        while(head != nullptr) {
            listnode.push_back(head);
            head = head->next;
        }
        int idx = listnode.size() - n;
        if (listnode.size() == 1) {
            assert(idx == 0);
            return nullptr;
        }
        if (idx == 0) {
            delete listnode[idx];
            return listnode[idx+1];
        }
        ListNode *before = listnode[idx-1];
        before->next = listnode[idx]->next;
        delete listnode[idx];
        return listnode[0];
    }
};
```

---

## 20. [有效的括号](https://leetcode.cn/problems/valid-parentheses/description/?envType=study-plan-v2&envId=top-100-liked)

用栈来判断。

```cpp
class Solution {
public:
    bool isValid(string s) {
        vector<int> sta; sta.reserve(s.size());
        for (int i = 0; i < s.size(); ++ i) {
            if (s[i] == ')' || s[i] == '}' || s[i] == ']') {
                if (s[i] == ')') {
                    if (sta.empty() || sta.back() != '(') return false;
                    else sta.pop_back();
                } else if (s[i] == '}') {
                    if (sta.empty() || sta.back() != '{') return false;
                    else sta.pop_back();
                } else {
                    if (sta.empty() || sta.back() != '[') return false;
                    else sta.pop_back();
                }
            } else sta.push_back(s[i]);
        }
        return sta.empty();
    }
};
```

---

## 21. [合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/description/?envType=study-plan-v2&envId=top-100-liked)

直接合并

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode head, *p = &head;
        while(list1 != nullptr && list2 != nullptr) {
            if (list1->val < list2->val) {
                p->next = list1;
                p = p->next;
                list1 = list1->next;
            } else {
                p->next = list2;
                p = p->next;
                list2 = list2->next;
            }
        }
        while(list1 != nullptr) p->next = list1, p = p->next, list1 = list1->next;
        while(list2 != nullptr) p->next = list2, p = p->next, list2 = list2->next;
        return head.next;
    }
};
```

---

## 22. [括号生成](https://leetcode.cn/problems/generate-parentheses/description/?envType=study-plan-v2&envId=top-100-liked)

可以用递归来做，做的同时还可以剪枝，会快一些。我为了方便直接写了迭代枚举+判断的方法。

```cpp
class Solution {
    bool check(int val, int len, string &ret) {
        ret = "";
        int leftNumber = 0;
        for (int i = 0; i < len; ++ i) {
            if (val & (1 << i)) {
                ret += '(';
                ++ leftNumber;
            } else {
                if (leftNumber) --leftNumber;
                else return false;
                ret += ')';
            }
        }
        return leftNumber == 0;
    }
public:
    vector<string> generateParenthesis(int n) {
        vector<string> ans;
        int len = n << 1;
        for (int i = 0; i < (1 << len); ++ i) {
            string ret;
            if (check(i, len, ret)) ans.push_back(ret);
        }
        return ans;
    }
};
```

---

## 23. [合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/description/?envType=study-plan-v2&envId=top-100-liked)

建一个堆，然后保存每个链表的当前指针，每次取出堆顶元素，然后将堆顶元素的下一个元素插入堆中。

```cpp
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode *head = nullptr;
        ListNode *tail = nullptr;
        priority_queue <pair<int, int>,vector<pair<int, int>>,greater<pair<int, int>> > heap;
        for (int i = 0; i < lists.size(); ++ i) {
            if (lists[i] != nullptr)
                heap.push(make_pair(lists[i]->val, i));
        }
        while(!heap.empty()) {
            pair<int, int> ret = heap.top();
            heap.pop();
            ListNode *nw = lists[ret.second];
            lists[ret.second] = nw->next;
            nw->next = nullptr;
            if (head == nullptr) {
                head = nw;
                tail = nw;
            } else {
                tail->next = nw;
                tail = nw;
            }
            if (lists[ret.second] != nullptr) {
                heap.push(make_pair(lists[ret.second]->val, ret.second));
            }
        }
        return head;
    }
};
```

---

## 24. [两两交换链表中的节点](https://leetcode.cn/problems/swap-nodes-in-pairs/description/?envType=study-plan-v2&envId=top-100-liked)

创建一个 `preHead` 作为头节点，来方便后续的判断。

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode *p, *q;
        ListNode preHead; preHead.next = head;
        if (head == nullptr || head->next == nullptr) return head;
        p = &preHead;
        while(p->next != nullptr && p->next->next != nullptr) {
            q = p->next;
            ListNode *x = q->next;
            p->next = x;
            q->next = x->next;
            x->next = q;
            p = q;
        }
        return preHead.next;
    }
};
```

---

## 25. [K 个一组翻转链表](https://leetcode.cn/problems/reverse-nodes-in-k-group/description/?envType=study-plan-v2&envId=top-100-liked)

添加一个 leftGuard 节点指向 head，后续用两个指针表示 `(]` 的区间然后进行翻转。

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (k == 1) return head;
        ListNode *left, *right;
        ListNode leftGuard(0, head);
        right = left = &leftGuard;
        int cnt = 0;
        while(right->next != nullptr) {
            right = right->next;
            if (++cnt == k) {
                ListNode *now = left->next, *tail = left->next;
                ListNode *nextNode = now->next;
                while(nextNode != right) {
                    ListNode *tmp = nextNode->next;
                    nextNode->next = now;
                    now = nextNode;
                    nextNode = tmp;
                }
                left->next->next = right->next;
                right->next = now;
                left->next = right;
                left = right = tail;
                cnt = 0;
            }
        }
        return leftGuard.next;
    }
};
```

---

## 31. [下一个排列](https://leetcode.cn/problems/next-permutation/description/?envType=study-plan-v2&envId=top-100-liked)

从后往前找到第一个非降序的数字，然后将这个数字与后面的数字中第一个比它大的数字交换，然后将后面的数字翻转（调为正序）。

```cpp
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int i = nums.size() - 2;
        while(i >= 0 && nums[i] >= nums[i+1]) -- i;
        if (i >= 0) {
            int j = nums.size() - 1;
            while(j >= 0 && nums[i] >= nums[j]) -- j;
            swap(nums[i], nums[j]);
        }
        reverse(nums.begin() + i + 1, nums.end());
    }
};
```

---

## 32. [最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/description/?envType=study-plan-v2&envId=top-100-liked)

栈求解括号序列，用栈维护每一个左括号的位置，通过记录`f[i]`表示以第 `i` 个字符结尾的最长有效括号的长度。

```cpp
class Solution {
public:
    int longestValidParentheses(string s) {
        int ans = 0;
        stack<int> sta;
        vector<int> f(s.size());
        for (int i = 0, res = 0; i < s.size(); ++ i) {
            f[i] = 0;
            if (s[i] == '(') {
                sta.push(i);
            } else {
                if (sta.empty()) continue;
                else {
                    int idx = sta.top(); sta.pop();
                    if (idx == 0) f[i] = i - idx + 1;
                    else f[i] = f[idx-1] + (i - idx + 1);
                    ans = max(ans, f[i]);
                }
            }
        }
        return ans;
    }
};
```

---

## 33. [搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked)

通过二分确认分界点，然后进行分别二分查找。

```cpp
class Solution {
public:
    int binary_search(vector<int>& nums, int l, int r, int target) {
        while(l <= r) {
            int mid = l+r >> 1;
            if (nums[mid] < target) l = mid+1;
            else if (nums[mid] > target) r = mid-1;
            else return mid;
        }
        return -1;
    }
    int search(vector<int>& nums, int target) {
        int base = nums[0];

        if (nums.size() == 1) return target == base ? 0 : -1;

        int l = 1, r = nums.size() - 1, ans = 0;
        while (l <= r) {
            int mid = l+r >> 1;
            if (nums[mid] > base) l = mid+1;
            else {
                r = mid-1;
                ans = mid;
            }
        }
        int x = binary_search(nums, 0, ans-1, target);
        int y = binary_search(nums, ans, nums.size()-1, target);
        if (x == -1 && y == -1) return -1;
        return x == -1 ? y : x;
    }
};
```

---

## 34. [在排序数组中查找元素的第一个和最后一个位置](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked)

`upper_bound` 大法好。

```cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int p = upper_bound(nums.begin(), nums.end(), target-1) - nums.begin();
        int q = upper_bound(nums.begin(), nums.end(), target) - nums.begin();
        if (p == q) return vector<int>(2, -1);
        return vector<int>({p, q-1});
    }
};
```

---

## 35. [搜索插入位置](https://leetcode.cn/problems/search-insert-position/description/?envType=study-plan-v2&envId=top-100-liked)

`lower_bound` 大法好。

```cpp
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        return lower_bound(nums.begin(), nums.end(), target) - nums.begin();
    }
};
```

---

## 39. [组合总和](https://leetcode.cn/problems/combination-sum/description/?envType=study-plan-v2&envId=top-100-liked)

递归暴搜，因为要输出答案。

```cpp
class Solution {
    vector<vector<int>> ans;
    vector<int> ret;
    void dfs(vector<int> &vec, int idx, int val) {
        if (val == 0) { ans.push_back(ret); return; }
        if (idx >= vec.size()) return;
        dfs(vec, idx+1, val);
        int cnt = 1;
        for (; val - cnt*vec[idx] >= 0; ++ cnt) {
            ret.push_back(vec[idx]);
            dfs(vec, idx+1, val - cnt*vec[idx]);
        }
        while(--cnt) ret.pop_back();
    }
public:
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        ans.clear();
        dfs(candidates, 0, target);
        return ans;
    }
};
```

---

## 41. [缺失的第一个正数](https://leetcode.cn/problems/first-missing-positive/description/?envType=study-plan-v2&envId=top-100-liked)

利用原本的数组元素存储信息，每次遇到一个下标为 `i` 的数字 `num[i]` 时，通过不断的 `swap` 操作将 `num[i]` 放到正确的位置上，最后再遍历一次数组。

```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        for (int i = 0; i < nums.size(); ++ i) {
            while (nums[i] > 0 && nums[i] <= nums.size()) {
                if (nums[i] == nums[nums[i]-1]) break;
                swap(nums[i], nums[nums[i]-1]);
            }
        }
        for (int i = 0; i < nums.size(); ++ i) {
            if (nums[i] != i+1) return i+1;
        }
        return nums.size() + 1;
    }
};
```

---

## 42. [接雨水](https://leetcode.cn/problems/trapping-rain-water/?envType=study-plan-v2&envId=top-100-liked)

可以看到，在从左向右移动的过程中，每次最高点更新时就会有一个新的接雨水的坑出现。所以循环两次，第一次从左向右，第二次从右向左，如果最高点有更新的话就是出现了新的水坑。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int maxHeightIdx = 0;
        int sum = 0, ans = 0;
        for (int i = 1; i < height.size(); ++ i) {
            if (height[i] >= height[maxHeightIdx]) {
                ans += height[maxHeightIdx] * (i - maxHeightIdx - 1) - sum;
                sum = 0;
                maxHeightIdx = i;
            } else {
                sum += height[i];
            }
        }
        int middle = maxHeightIdx;
        maxHeightIdx = height.size()-1;
        sum = 0;
        for (int i = height.size()-2; i >= middle; -- i) {
            if (height[i] >= height[maxHeightIdx]) {
                ans += height[maxHeightIdx] * (maxHeightIdx - i - 1) - sum;
                sum = 0;
                maxHeightIdx = i;
            } else {
                sum += height[i];
            }
        }
        return ans;
    }
};
```

---

## 45. [跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/description/?envType=study-plan-v2&envId=top-100-liked)

每次选择跳跃区间内下一步跳的最远的点。迭代跳跃直到达到终点。这份代码其实复杂度有些不对，每次枚举不应该从 `cur` 开始，最好是从上一次的 `jumpRange` 开始，这样才可以避免枚举到相同的元素，但是懒得改了）

```cpp
class Solution {
public:
    int jump(vector<int>& nums) {
        if (nums.size() <= 1) return 0;
        int ans = 0, cur = 0, jumpRange = nums[0];
        while(true) {
            if (jumpRange >= nums.size()-1) break;
            int idx = cur;
            for (int i = cur + 1; i <= jumpRange; ++ i) {
                if (i + nums[i] > idx + nums[idx]) {
                    idx = i;
                }
            }
            jumpRange = idx + nums[idx];
            cur = idx;
            ans += 1;
        }
        ans += 1;
        return ans;
    }
};
```

---

## 46. [全排列](https://leetcode.cn/problems/permutations/?envType=study-plan-v2&envId=top-100-liked)

不断求下一个排列，直到得到本身。

如何求下一个排列：从后往前找到第一个不是递增的数字，然后将该数字后面的数字翻转。再从当前数字开始向后枚举找到第一个大于该数字的数字，交换两个数字的位置，就可以求出下一个排列。

```cpp
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> ans; ans.push_back(nums);
        vector<int> origin = nums;
        while(true) {
            int p = -1;
            for (int i = nums.size() - 2; i >= 0; -- i) {
                if (nums[i] < nums[i+1]) {
                    p = i; break;
                }
            }
            reverse(nums.begin()+p+1, nums.end());
            if (p != -1) for (int i = p+1; i < nums.size(); ++ i) {
                if (nums[p] < nums[i]) {
                    swap(nums[p], nums[i]);
                    break;
                }
            }
            if (nums == origin) break;
            ans.push_back(nums);
        }
        return ans;
    }
};
```

---

## 48. [旋转图像](https://leetcode.cn/problems/rotate-image/description/?envType=study-plan-v2&envId=top-100-liked)

旋转会形成循环链路，通过坐标计算可以计算出四个点的旋转变换关系，然后直接交换。注意对于奇数的矩阵来说需要枚举中间行或列来接触到中间的循环节。

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n>>1; ++ i) {
            for (int j = 0; j < (n+1>>1); ++ j) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n-j-1][i];
                matrix[n-j-1][i] = matrix[n-i-1][n-j-1];
                matrix[n-i-1][n-j-1] = matrix[j][n-i-1];
                matrix[j][n-i-1] = tmp;
            }
        }
    }
};
```

---

## 49. [字母异位词分组](https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-100-liked)

将所有字符串排序，此时相等的字符串就是字母异位词，用排序后的字符串作为 key，原字符串作为 value 存入 hashmap 中，最后将 hashmap 中的 value 取出。用 C++ 的 `unordered_map` 实现。

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        int len = strs.size();
        unordered_map<string, vector<string>> mp;
        int cnt = 0;
        for (auto it: strs) {
            string sortResult = it;
            sort(sortResult.begin(), sortResult.end());
            mp[sortResult].push_back(it);
        }
        vector<vector<string>> ret;
        for (auto it : mp) {
            ret.push_back(it.second);
        }
        return ret;
    }
};
```

---

## 51. [N 皇后](https://leetcode.cn/problems/n-queens/description/?envType=study-plan-v2&envId=top-100-liked)

暴搜

```cpp
class Solution {
    vector<vector<string>> ans;
    int mat[10][10];
public:
    vector<vector<string>> solveNQueens(int n) {
        ans.clear();
        memset(mat, 0, sizeof mat);
        dfs(n, 1, 1);
        return ans;
    }

    void dfs(int n, int rows, int idx) {
        if (idx == n+1) {
            return ;
        }
        if (rows > n) {
            vector<string> res;
            res.clear();
            for (int i = 1; i <= n; ++ i) {
                string row_ans = "";
                for (int j = 1; j <= n; ++ j) {
                    if (mat[i][j]) row_ans += "Q";
                    else row_ans += ".";
                }
                res.push_back(row_ans);
            }
            ans.push_back(res);
            return ;
        }
        if (check(n, rows, idx)) {
            mat[rows][idx] = 1;
            dfs(n, rows+1, 1);
            mat[rows][idx] = 0;
        }
        dfs(n, rows, idx+1);
    }

    bool check(int n, int x, int y) {
        for (int i = 1; x-i > 0 && y - i > 0; ++ i) {
            if (mat[x - i][y - i]) return false;
        }
        for (int i = 1; x-i > 0 && y + i <= n; ++ i) {
            if (mat[x - i][y + i]) return false;
        }
        for (int i = 1; i < x; ++ i) {
            if (mat[i][y]) return false;
        }
        return true;
    }
};
```

---

## 53. [最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2&envId=top-100-liked)

维护一个变量表示从当前数字往前扩展能得到的最大子数组和。

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int ans = -0x7fffffff;
        int res = -0x7fffffff;
        for (auto x: nums) {
            if (res < 0) {
                ans = max(ans, x);
                res = x;
            } else {
                ans = max(ans, res + x);
                res += x;
            }
        }
        return ans;
    }
};
```

---

## 54. [螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/description/?envType=study-plan-v2&envId=top-100-liked)

通过维护边界线的方式来完成，维护目前剩余的上下左右四个边界线。

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> ans;
        if (matrix.empty()) return ans;
        int u = 0, l = 0;
        int d = matrix.size() - 1, r = matrix[0].size() - 1;
        ans.reserve(matrix.size() * matrix[0].size());
        while(1) {
            for (int i = l; i <= r; ++ i) ans.push_back(matrix[u][i]);
            if (++ u > d) break;
            for (int i = u; i <= d; ++ i) ans.push_back(matrix[i][r]);
            if (-- r < l) break;
            for (int i = r; i >= l; -- i) ans.push_back(matrix[d][i]);
            if (-- d < u) break;
            for (int i = d; i >= u; -- i) ans.push_back(matrix[i][l]);
            if (++ l > r) break;
        }
        return ans;
    }
};
```

---

## 55. [跳跃游戏](https://leetcode.cn/problems/jump-game/description/?envType=study-plan-v2&envId=top-100-liked)

维护可以跳到的最远的点。

```cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        if (nums.size() <= 1) return true;
        for (int i = 0, idx = 0; i < nums.size() && i <= idx; ++ i) {
            while(idx < i + nums[i]) if (++idx >= nums.size()-1) return true;
        }
        return false;
    }
};
```

---

## 56. [合并区间](https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2&envId=top-100-liked)

按照左端点排序后线形扫一遍并进行合并

```cpp
class Solution {
    static bool compare(const vector<int> &a, const vector<int> &b) {
        return a[0] < b[0];
    }
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        vector<vector<int>> ans;
        if (intervals.empty()) return ans;
        sort(intervals.begin(), intervals.end(), compare);
        int l = intervals[0][0], r = intervals[0][1];
        for (int i = 1; i < intervals.size(); ++ i) {
            if (intervals[i][0] <= r) {
                r = max(r, intervals[i][1]);
            } else {
                ans.push_back(vector<int>({l, r}));
                l = intervals[i][0];
                r = intervals[i][1];
            }
        }
        ans.push_back(vector<int>({l, r}));
        return ans;
    }
};
```

---

## 62. [不同路径](https://leetcode.cn/problems/unique-paths/description/?envType=study-plan-v2&envId=top-100-liked)

`f[i][j]` 表示从起点走到 $(i, j)$ 的方案数: `f[i][j] = f[i-1][j] + f[i][j-1]`

```cpp
class Solution {
    int f[100][100];
public:
    int uniquePaths(int m, int n) {
        f[0][0] = 1;
        for (int i = 1; i < m; ++ i) f[i][0] = 1;
        for (int i = 1; i < n; ++ i) f[0][i] = 1;
        for (int i = 1;i < m; ++ i) {
            for (int j = 1; j < n; ++ j) {
                f[i][j] = f[i-1][j] + f[i][j-1];
            }
        }
        return f[m-1][n-1];
    }
};
```

---

## 64. [最小路径和](https://leetcode.cn/problems/minimum-path-sum/description/?envType=study-plan-v2&envId=top-100-liked)

动态规划，`f[i][j] = min(f[i-1][j], f[i][j-1]) + grid[i][j]`。

```cpp
class Solution {
    vector<vector<int>> f;
public:
    int minPathSum(vector<vector<int>>& grid) {
        f = grid;
        for (int i = 0; i < grid.size(); ++ i) {
            for (int j = 0; j < grid[i].size(); ++ j) {
                if (i == 0 && j == 0) continue;
                if (i == 0) f[i][j] = f[i][j-1] + grid[i][j];
                else if (j == 0) f[i][j] = f[i-1][j] + grid[i][j];
                else f[i][j] = min(f[i-1][j], f[i][j-1]) + grid[i][j];
            }
        }
        return f.back().back();
    }
};
```

---

## 70. [爬楼梯](https://leetcode.cn/problems/climbing-stairs/description/?envType=study-plan-v2&envId=top-100-liked)

记忆化搜索，斐波那契数列。

```cpp
class Solution {
    int f[55];

public:
    Solution() {
        memset(f, -1, sizeof f);
        f[0] = 1;
        f[1] = 1;
    }
    int climbStairs(int n) {
        if (f[n] != -1) return f[n];
        return f[n] = climbStairs(n-1) + climbStairs(n-2);
    }
};
```

---

## 72. [编辑距离](https://leetcode.cn/problems/edit-distance/description/?envType=study-plan-v2&envId=top-100-liked)

`f[i][j]` 表示 `s[0..i-1]` 和 `t[0..j-1]` 的编辑距离

```cpp
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size(), m = word2.size();
        vector<vector<int>> f(n+1, vector<int>(m+1));
        f[0][0] = 0;
        for (int i = 1; i <= n; ++ i) f[i][0] = i;
        for (int i = 1; i <= m; ++ i) f[0][i] = i;
        for (int i = 1; i <= n; ++ i) {
            for (int j = 1; j <= m; ++ j) {
                if (word1[i-1] == word2[j-1]) f[i][j] = f[i-1][j-1];
                else f[i][j] = min(min(f[i-1][j-1], f[i-1][j]), f[i][j-1]) + 1;
            }
        }
        return f[n][m];
    }
};
```

---

## 73. [矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

将行或者列的置零信号保存到第 $0$ 行和第 $0$ 列，然后再开两个变量分别表示第 $0$ 行和第 $0$ 列是否需要置零。

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        bool line0 = false, col0 = false;
        for (int i = 0; i < matrix[0].size(); ++ i) {
            if (matrix[0][i] == 0) {
                line0 = true;
                break;
            }
        }
        for (int i = 0; i < matrix.size(); ++ i) {
            if (matrix[i][0] == 0) {
                col0 = true;
            }
            for (int j = 0; j < matrix[i].size(); ++ j) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        for (int i = 1; i < matrix[0].size(); ++ i) {
            if (matrix[0][i] == 0) {
                for (int j = 1; j < matrix.size(); ++ j) {
                    matrix[j][i] = 0;
                }
            }
        }
        for (int i = 1; i < matrix.size(); ++ i) {
            if (matrix[i][0] == 0) {
                for (int j = 1; j < matrix[i].size(); ++ j) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (line0) for(int i = 0; i < matrix[0].size(); ++ i) matrix[0][i] = 0;
        if (col0) for(int i = 0; i < matrix.size(); ++ i) matrix[i][0] = 0;
    }
};
```

---

## 74. [搜索二维矩阵](https://leetcode.cn/problems/search-a-2d-matrix/description/?envType=study-plan-v2&envId=top-100-liked)

突出一个，懒得写二分）先用一次二分确定行坐标，再用一次二分确定列坐标。

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int p = lower_bound(matrix.begin(), matrix.end(), vector<int>(1, target)) - matrix.begin();

        if (p < matrix.size() && matrix[p][0] == target) return true;
        if (-- p < 0) return false;

        auto it = lower_bound(matrix[p].begin(), matrix[p].end(), target);
        if (it == matrix[p].end()) return false;
        return *it == target;
    }
};
```

---

## 75. [颜色分类](https://leetcode.cn/problems/sort-colors/description/?envType=study-plan-v2&envId=top-100-liked)

维护左指针表示 $0$ 延续到哪里了，同时维护右指针表示 $2$ 延续到哪里了。从左边换过来的只可能是 $2$，所以可以直接下一个，从右边换过来的不确定，所以需要 `while` 循环。代码上我是都放到了 `while` 里面，但其实从左边换过来的可以提到循环外面。

```cpp
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int l = 0, r = nums.size()-1;
        for (int i = 0; i <= r; ++ i) {
            while(l <= i && i <= r) {
                if (nums[i] == 0) swap(nums[l++], nums[i]);
                else if (nums[i] == 2) swap(nums[r--], nums[i]);
                else break;
            }
        }
        return ;
    }
};
```

---

## 76. [最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/description/?envType=study-plan-v2&envId=top-100-liked)

枚举右端点，左端点一定是一直向右移动的。

```cpp
class Solution {
    int ws[52], nw[52], remain;
    int convertChar2Id(char x) {
        if ('a' <= x && x <= 'z') return x - 'a';
        if ('A' <= x && x <= 'Z') return x - 'A' + 26;
        return -1;
    }
public:
    string minWindow(string s, string t) {
        memset(ws, 0, sizeof ws);
        memset(nw, 0, sizeof nw);
        remain = 52;
        int n = s.size(), m = t.size();
        for (int i = 0; i < m; ++ i) {
            ws[convertChar2Id(t[i])] += 1;
        }
        for (int i = 0; i < 52; ++ i) {
            if (nw[i] >= ws[i]) -- remain;
        }
        int ans = n+1, ansl = -1, ansr = -1;
        for (int r = 0,l = 0; r < n; ++ r) {
            int charId = convertChar2Id(s[r]);
            if (++ nw[charId] == ws[charId]) {
                -- remain;
            }
            while(l < r) {
                charId = convertChar2Id(s[l]);
                if (nw[charId] > ws[charId]) {
                    -- nw[charId];
                    ++ l;
                } else break;
            }
            if (remain == 0){
                if (r-l+1 < ans) {
                    ans = r-l+1;
                    ansl = l;
                    ansr = r;
                }
            }
        }
        if (ans == n+1) return "";
        return s.substr(ansl, ansr-ansl+1);
    }
};
```

---

## 79. [子集](https://leetcode.cn/problems/subsets/description/?envType=study-plan-v2&envId=top-100-liked)

二进制枚举

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        for (int i = 0 ; i < (1 << nums.size()); ++ i) {
            vector<int> res;
            for (int j = 0; j < nums.size(); ++ j) {
                if (i & (1 << j)) res.push_back(nums[j]);
            }
            ans.push_back(res);
        }
        return ans;
    }
};
```

---

## 79. [单词搜索](https://leetcode.cn/problems/word-search/description/?envType=study-plan-v2&envId=top-100-liked)

直接搜索，记录当前位置匹配到了第几个元素，然后递归搜索。

```cpp
class Solution {
    bool flag[12][12];
    bool dfs(vector<vector<char>>& board, int u, int v, string word, int idx) {
        static int dx[] = {0, 0, 1, -1};
        static int dy[] = {1, -1, 0, 0};
        if (flag[u][v]) return false;
        flag[u][v] = true;
        if (board[u][v] != word[idx]) {
            flag[u][v] = false;
            return false;
        }

        if (idx == word.size()-1) {
            flag[u][v] = false;
            return true;
        }
        for (int i = 0; i < 4; ++ i) {
            int nx = u + dx[i];
            int ny = v + dy[i];
            if (nx < 0 || nx >= board.size() || ny < 0 || ny >= board[0].size()) continue;
            if (dfs(board, nx, ny, word, idx+1)) {
                flag[u][v] = false;
                return true;
            }
        }
        flag[u][v] = false;
        return false;
    }
public:
    bool exist(vector<vector<char>>& board, string word) {
        for (int i = 0; i < board.size(); ++ i) {
            for (int j = 0;j < board[i].size(); ++ j) {
                if (dfs(board, i, j, word, 0)) return true;
            }
        }
        return false;
    }
};
```

---

## 84. [柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/description/?envType=study-plan-v2&envId=top-100-liked)

每一个元素 $i$ 向左能扩展的最大长度可以通过一个单调上升的单调栈计算得到，两边单调栈求出每一个元素向左扩展和向右扩展的最大长度，然后计算答案。

```cpp
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        stack<int> sta;
        vector<int> left(heights.size()), right(heights.size());
        for (int i = 0; i < heights.size(); ++ i) {
            while(!sta.empty() && heights[sta.top()] > heights[i]) sta.pop();
            if (sta.empty()) left[i] = 0;
            else {
                if (heights[sta.top()] == heights[i]) left[i] = left[sta.top()];
                else left[i] = sta.top() + 1;
            }
            sta.push(i);
        }
        while(!sta.empty()) sta.pop();
        int ans = 0;
        for (int i = heights.size()-1; i >= 0; -- i) {
            while(!sta.empty() && heights[sta.top()] > heights[i]) sta.pop();
            if (sta.empty()) right[i] = heights.size()-1;
            else {
                if (heights[sta.top()] == heights[i]) right[i] = right[sta.top()];
                else right[i] = sta.top() - 1;
            }
            sta.push(i);
            ans = max(ans, (i-left[i] + right[i]-i + 1) * heights[i]);
            sta.push(i);
        }
        return ans;
    }
};
```

---

## 94. [二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/description/?envType=study-plan-v2&envId=top-100-liked)

```cpp
class Solution {
    void dfs(TreeNode *root, vector<int> &ans) {
        if (root == nullptr) return ;
        dfs(root->left, ans);
        ans.push_back(root->val);
        dfs(root->right, ans);
        return ;
    }
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> ans;
        dfs(root, ans);
        return ans;
    }
};
```

---

## 98. [验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/description/?envType=study-plan-v2&envId=top-100-liked)

卡 `INT_MAX` 很有意思吗？

```cpp
class Solution {
    bool dfs(TreeNode *root, long long minVal, long long maxVal) {
        if (root == nullptr) return true;
        if (root->val <= minVal || root->val >= maxVal) return false;
        return dfs(root->left, minVal, root->val) && dfs(root->right, root->val, maxVal);
    }
public:
    bool isValidBST(TreeNode* root) {
        return dfs(root, -(1LL<<60), (1LL<<60));
    }
};
```

---

## 101. [对称二叉树](https://leetcode.cn/problems/symmetric-tree/description/?envType=study-plan-v2&envId=top-100-liked)

递归判断左右子树是否对称。

```cpp
class Solution {
    bool check(TreeNode *p, TreeNode *q) {
        if (p == nullptr && q == nullptr) return true;
        if (p == nullptr || q == nullptr) return false;
        if (p->val != q->val) return false;
        return check(p->left, q->right) && check(p->right, q->left);
    }
public:
    bool isSymmetric(TreeNode* root) {
        TreeNode *p = root, *q = root;
        return check(p, q);
    }
};
```

---

## 102. [二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/?envType=study-plan-v2&envId=top-100-liked)

BFS + 守卫节点

```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> ans;
        if (root == nullptr) return ans;
        queue<TreeNode*> que; que.push(root); que.push(nullptr);
        vector<int> ret;
        while(!que.empty()) {
            TreeNode *p = que.front(); que.pop();
            if (p == nullptr) {
                ans.push_back(ret);
                ret.clear();
                if (que.empty()) break;
                else {
                    que.push(nullptr);
                    continue;
                }
            }
            ret.push_back(p->val);
            if (p->left != nullptr) que.push(p->left);
            if (p->right!= nullptr) que.push(p->right);
        }
        return ans;
    }
};
```

---

## 104. [二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked)

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr) return 0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
```

---

## 105. [从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/?envType=study-plan-v2&envId=top-100-liked)

以前序遍历为主，每次找到当前位置在中序遍历中出现的位置来将问题划分为两个子问题。

```cpp
class Solution {
    TreeNode* dfs(vector<int>& preorder, vector<int> &inorder, int &idx, int l, int r) {
        if (l > r) return nullptr;
        TreeNode *p = new TreeNode();
        p->val = preorder[idx];
        int mid = -1;
        for (int i = l; i <= r; ++ i) {
            if (inorder[i] == preorder[idx]) {
                mid = i;
                break;
            }
        }
        assert(mid != -1);
        idx += 1;

        p->left = dfs(preorder, inorder, idx, l, mid-1);
        p->right = dfs(preorder, inorder, idx, mid+1, r);
        return p;
    }
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int idx = 0;
        return dfs(preorder, inorder, idx, 0, inorder.size()-1);
    }
};
```

---

## 108. [将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description/?envType=study-plan-v2&envId=top-100-liked)

通过 dfs 进行递归建树，每次取中间的数字作为根节点，然后递归建立左右子树。

```cpp
class Solution {
public:
    TreeNode* dfs(int l, int r, vector<int>& nums) {
        if (l > r) return NULL;
        int mid = l+r >> 1;
        TreeNode* rt = new TreeNode();
        rt->val = nums[mid];
        rt->left = dfs(l, mid-1, nums);
        rt->right = dfs(mid+1, r, nums);
        return rt;
    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return dfs(0, nums.size()-1, nums);
    }
};
```

---

## 114. [二叉树展开为链表](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/?envType=study-plan-v2&envId=top-100-liked)

可以通过递归的方式来做，每次返回链表的头与尾部。

```cpp
class Solution {
    pair<TreeNode*, TreeNode*> dfs(TreeNode *p) {
        if (p == nullptr) return make_pair(nullptr, nullptr);
        auto listleft = dfs(p->left);
        auto listRight = dfs(p->right);
        p->left = nullptr;
        if (listleft.first == nullptr) p->right = listRight.first;
        else {
            p->right = listleft.first;
            (listleft.second)->right = listRight.first;
        }
        if (p->right == nullptr) return make_pair(p, p);
        if (listRight.first == nullptr) return make_pair(p, listleft.second);
        return make_pair(p, listRight.second);
    }
public:
    void flatten(TreeNode* root) {
        dfs(root);
    }
};
```

看了题解之后发现也可以通过迭代的方式来做，每次将左子树插到右子树中，这种方式要慢一些，但是空间复杂度降为了 $O(1)$。

```cpp
class Solution {
public:
    void flatten(TreeNode* root) {
        while(root != nullptr) {
            if (root->left != nullptr) {
                auto tmp = root->left;
                while(tmp->right != nullptr) tmp = tmp->right;
                tmp->right = root->right;
                root->right = root->left;
                root->left = nullptr;
            }
            root = root->right;
        }
    }
};
```

---

## 118. [杨辉三角](https://leetcode.cn/problems/pascals-triangle/description/?envType=study-plan-v2&envId=top-100-liked)

```cpp
class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> f(numRows);
        f[0].push_back(1);
        for (int i = 1; i < numRows; ++ i) {
            f[i].resize(i+1);
            for (int j = 0; j <= i; ++ j) {
                if (j != i) f[i][j] += f[i-1][j];
                if (j != 0) f[i][j] += f[i-1][j-1];
            }
        }
        return f;
    }
};
```

---

## 121. [买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/?envType=study-plan-v2&envId=top-100-liked)

记录过去最低的买入价格。

```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.size() <= 1) return 0;
        int minVal = prices[0], ans = 0;
        for (int i = 1;i < prices.size(); ++ i) {
            if (prices[i] > minVal) ans = max(ans, prices[i] - minVal);
            else minVal = prices[i];
        }
        return ans;
    }
};
```

---

## 124. [二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/?envType=study-plan-v2&envId=top-100-liked)

在每个点上计算以这个点为根的子树中端点为根的最大路径，因为是二叉树，所以每次统计答案的时候直接 `left + right` 就是通过根的最大路径。

```cpp
class Solution {
    int ans;
    int dfs(TreeNode *p) {
        if (p == nullptr) return 0;
        int left = max(dfs(p->left), 0);
        int right = max(dfs(p->right), 0);
        ans = max(ans, left + right + p->val);
        return max(left, right) + p->val;;
    }
public:
    int maxPathSum(TreeNode* root) {
        ans = -0x7fffffff;
        dfs(root);
        return ans;
    }
};
```

---

## 128. [最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/?envType=study-plan-v2&envId=top-100-liked)

将所有数字插入到 HashMap 内后，通过枚举的方式寻找连续段的起点，然后通过循环找到终点。

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> num_set;
        for (auto x: nums) {
            num_set.insert(x);
        }
        int ans = 0;
        for (auto x: num_set) {
            if (!num_set.count(x-1)) {
                int cur = x;
                int ret = 1;
                while(num_set.count(cur+1)) {
                    ++ cur;
                    ++ ret;
                }
                ans = max(ans, ret);
            }
        }
        return ans;
    }
};
```

---

## 131. [分割回文串](https://leetcode.cn/problems/palindrome-partitioning/?envType=study-plan-v2&envId=top-100-liked)

搜索枚举：

```cpp
class Solution {
    vector<string> res;
    vector<vector<string>> ans;
    bool check(string s) {
        for (int i = 0; i < s.size() / 2; ++ i) {
            if (s[i] != s[s.size()-i-1]) return false;
        }
        return true;
    }
    void dfs(int u, string s) {
        if (u >= s.size()) {
            ans.push_back(res);
            return ;
        }
        for (int i = u;i < s.size(); ++ i) {
            string str = s.substr(u, i - u + 1);
            if (check(str)) {
                res.push_back(str);
                dfs(i+1, s);
                res.pop_back();
            }
        }
    }
public:
    vector<vector<string>> partition(string s) {
        res.clear();
        ans.clear();
        dfs(0, s);
        return ans;
    }
};
```

---

## 138. [随机链表的复制](https://leetcode.cn/problems/copy-list-with-random-pointer/description/?envType=study-plan-v2&envId=top-100-liked)

用一个 HashMap 来记录下来每一个节点对应的新节点，在拷贝时通过记忆化的方式来将指针指向正确的位置。

```cpp
class Solution {
    unordered_map<Node*, Node*> hash;
    Node* copy(Node *head) {
        if (head == nullptr) return nullptr;
        if (hash.contains(head)) return hash[head];
        Node *p = new Node(head->val);
        hash[head] = p;
        p->next = copy(head->next);
        p->random = copy(head->random);
        return p;
    }
public:
    Node* copyRandomList(Node* head) {
        hash.clear();
        return copy(head);
    }
};
```

---

## 136. [只出现一次的数字](https://leetcode.cn/problems/single-number/description/?envType=study-plan-v2&envId=top-100-liked)

异或和，出现偶数次的数字都会互相抵消。

```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int x = 0;
        for (int i = 0; i < nums.size(); ++ i) {
            x ^= nums[i];
        }
        return x;
    }
};
```

---

## 139. [单词拆分](https://leetcode.cn/problems/word-break/description/?envType=study-plan-v2&envId=top-100-liked)

设 `f[i]` 表示能够凑出 `0 .. i-1`, 枚举转移更新。判断可以用 Hash 表或者 Trie。

```cpp
class Solution {
    vector<bool> f;
    unordered_set<string> hash;
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        hash.clear();
        for (auto str : wordDict) hash.insert(str);
        f.clear(); f.resize(s.size()+1);
        f[0] = true;
        for (int i = 1; i <= s.size(); ++ i) {
            for (int j = 0; j < i; ++ j) {
                if (f[j] && hash.contains(s.substr(j, i-j))) f[i] = true;
            }
        }
        return f[s.size()];
    }
};
```

---

## 141. [环形链表](https://leetcode.cn/problems/linked-list-cycle/description/?envType=study-plan-v2&envId=top-100-liked)

我居然先做的*环形链表 II*。

```cpp
class Solution {
#define next(x) \
    if((x)->next == nullptr) return false; \
    else (x) = (x)->next
public:
    bool hasCycle(ListNode *head) {
        if (head == nullptr) return false;
        ListNode *p = head, *q = head;
        while(true) {
            next(p);
            next(q);next(q);
            if (p == q) return true;
        }
    }
#undef next(x)
};
```

---

## 143. [环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/description/?envType=study-plan-v2&envId=top-100-liked)

首先在 Head 定义快慢两个指针，快指针每次移动 $2$，慢指针每次移动 $1$。如果我们设 $L$ 表示入口点到入环点的距离，$R$ 表示环的长度，那么我们可以计算出慢指针到达入环点时快指针位于环上距离入环点 $L$ 的位置，所以再经过 $(R-L)$ 步就会到达相遇点，所以相遇点位于环上距离入环点 $(R-L)$ 步的位置。

接下来将慢指针重新放回起点，然后将快指针的步幅调整为 $1$。这样在经过 $L$ 步后，慢指针刚好到达入环点，快指针也刚好到达入环点。因此此时的相遇点就是入环点，也就是题目所求的位置。

```cpp
#define next(pointer) \
    if ((pointer)->next != nullptr) pointer = pointer->next; \
    else return nullptr

class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if (head == nullptr) return nullptr;
        ListNode *slowPointer = head, *fastPointer = head;
        do {
            next(slowPointer);
            next(fastPointer);
            next(fastPointer);
        } while(slowPointer != fastPointer);
        slowPointer = head;
        while(slowPointer != fastPointer) {
            next(slowPointer);
            next(fastPointer);
        }
        return slowPointer;
    }
};
#undef next(pointer)
```

---

## 146. [LRU 缓存](https://leetcode.cn/problems/lru-cache/?envType=study-plan-v2&envId=top-100-liked)

hash 表维护每个 key 对应的指针，双向链表维护数据的顺序（强烈建议使用头尾实节点来写）。

```cpp
struct Node {
    int key, value;
    Node *nxt, *pre;
};

class LRUCache {
    unordered_map<int, Node*> hash;
    int cap, siz;
    Node *head, *tail;
public:
    LRUCache(int capacity) {
        cap = capacity;
        siz = 0;
        hash.clear();
        head = new Node();
        tail = new Node();
        head->key = head->value = -1;
        head->pre = NULL;
        head->nxt = tail;

        tail->key = tail->value = -1;
        tail->pre = head;
        tail->nxt = NULL;
    }

    int get(int key) {
        if (hash.contains(key)) {
            Node* node = hash[key];
            update(node, node->value);
            return node->value;
        }
        return -1;
    }

    void put(int key, int value) {
        if (hash.contains(key)) {
            Node* node = hash[key];
            update(hash[key], value);
        } else {
            hash[key] = insert(key, value);
            if (++siz > cap) {
                int delete_key = delete_last();
                hash.erase(delete_key);
            }
        }
    }

    Node* insert(int key, int value) {
        Node *ret = new Node;
        ret->value = value;
        ret->key = key;
        ret->nxt = ret->pre = NULL;

        Node* tmp = head->nxt;
        head->nxt = tmp->pre = ret;
        ret->pre = head;
        ret->nxt = tmp;

        return ret;
    }

    void update(Node *p, int value) {
        p->value = value;

        p->pre->nxt = p->nxt;
        p->nxt->pre = p->pre;

        p->pre = head;
        p->nxt = head->nxt;

        head->nxt->pre = p;
        head->nxt = p;
    }

    int delete_last() {
        Node *last = tail->pre;

        last->pre->nxt = last->nxt;
        last->nxt->pre = last->pre;

        int ret = last->key;
        delete last;

        return ret;
    }

};
```

---

## 148. [排序链表](https://leetcode.cn/problems/sort-list/description/?envType=study-plan-v2&envId=top-100-liked)

归并排序，通过大步小步算法找到链表的中间点，然后划分为两段进行归并排序。

```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if (head == nullptr) return nullptr;
        if (head->next == nullptr) return head;
        ListNode *p = head, *q = head->next;
        while(q != nullptr) {
            q = q->next; if (q == nullptr) break;
            q = q->next;
            p = p->next;
        }
        ListNode *list1 = sortList(p->next); p->next = nullptr;
        ListNode *list2 = sortList(head);
        ListNode preHead; head = &preHead;
        while(list1 != nullptr && list2 != nullptr) {
            if (list1->val < list2->val) {
                head->next = list1;
                list1 = list1->next;
                head = head->next;
            } else {
                head->next = list2;
                list2 = list2->next;
                head = head->next;
            }
        }
        while(list1 != nullptr) {
            head->next = list1; list1 = list1->next; head = head->next;
        }
        while(list2 != nullptr) {
            head->next = list2; list2 = list2->next; head = head->next;
        }
        return preHead.next;
    }
};
```

---

## 152. [乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/description/?envType=study-plan-v2&envId=top-100-liked)

> 我不喜欢这个题目，出题人玩文字游戏。保证了答案不超过`int`，但是在数据中刻意让运算过程的数字爆掉了`long long`。与约定俗成的习惯不符。

维护`minDot`和`maxDot`两个变量，保存选择当前数字作为结尾的情况下，能获得的最大乘积和最小乘积分别是多少。

```cpp
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        long long maxDot = nums[0];
        long long minDot = nums[0];
        long long ans = nums[0];
        for (int i = 1; i < nums.size(); ++ i) {
            long long x = nums[i];
            if (x > 0) {
                ans = max(ans, max(maxDot * x, x));
                maxDot = max(x, maxDot * x);
                minDot = min((long long)x, minDot * x);
                if (minDot < -0x7fffffff) minDot = -0x7fffffff;
            } else {
                ans = max(ans, max(minDot * x, x));
                long long tmp = maxDot;
                maxDot = max(x, minDot * x);
                minDot = min((long long)x, tmp * x);
                if (minDot < -0x7fffffff) minDot = -0x7fffffff;
            }
        }
        return ans;
    }
};
```

---

## 153. [寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/?envType=study-plan-v2&envId=top-100-liked)

如果我们取第一个数字，那么所有大于这个数字的数分布在数组的左半边，小于这个数字的数分布在数组的右半边，通过二分确认分界点。

```cpp
class Solution {
public:
    int findMin(vector<int>& nums) {
        int midValue = nums[0];
        int l = 1, r = nums.size() - 1, ans = -1;
        while(l <= r) {
            int mid = l+r >> 1;
            if (nums[mid] < midValue) {
                r = mid-1;
                ans = mid;
            } else l = mid+1;
        }
        if (ans == -1) return nums[0];
        return nums[ans];
    }
};
```

---

## 155. [最小栈](https://leetcode.cn/problems/min-stack/description/?envType=study-plan-v2&envId=top-100-liked)

用了两个数组，一个数组维护栈，另一个数组维护单调栈。因为更靠前更小的数字可以阻止后面所有比他大的数字成为最小值。

```cpp
class MinStack {
    vector<int> sta;
    vector<int> minStack;
public:
    MinStack() {
        sta.clear();
        minStack.clear();
    }

    void push(int val) {
        sta.push_back(val);
        if(minStack.empty() || minStack.back() >= val)
            minStack.push_back(val);
    }

    void pop() {
        if (minStack.back() == sta.back()) {
            minStack.pop_back();
            sta.pop_back();
        } else sta.pop_back();
    }

    int top() {
        return sta.back();
    }

    int getMin() {
        return minStack.back();
    }
};
```

---

### 160. [相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2&envId=top-100-liked)

同时开始遍历，较短的一方遍历完之后可以得知长的一条路线比短的要多多少。可以让短的直接从长的头开始遍历，等长的头到终点后前往短的头开始遍历，这样双方会同时抵达。关于不相交的链表：我们可以理解为链表的末尾都连向了 `nullptr`，所以如果两个链表不相交的话最终会在 `nullptr` 相遇。

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) return nullptr;
        ListNode *pA = headA, *pB = headB;
        while(pA != pB) {
            pA = pA == nullptr ? headB : pA->next;
            pB = pB == nullptr ? headA : pB->next;
        }
        return pA;
    }
};
```

---

## 162. [多数元素](https://leetcode.cn/problems/majority-element/description/?envType=study-plan-v2&envId=top-100-liked)

维护目前的主元素和该元素出现的次数，任意与该元素不同的数字视作**抵消**。

```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int value = -1, cnt = 0;
        for (auto x : nums) {
            if (x != value) {
                if (cnt) -- cnt;
                else if (cnt == 0) {
                    value = x;
                    ++ cnt;
                }
            } else ++ cnt;
        }
        return value;
    }
};
```

---

## 189. [轮转数组](https://leetcode.cn/problems/rotate-array/description/)

十分有想象力的做法：先整体翻转，再分两部分翻转回来。

```cpp
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k %= nums.size();
        reverse(nums.begin(), nums.end());
        reverse(nums.begin(), nums.begin() + k);
        reverse(nums.begin()+k, nums.end());
    }
};
```

---

## 198. [打家劫舍](https://leetcode.cn/problems/house-robber/description/?envType=study-plan-v2&envId=top-100-liked)

动态规划，`f[i]` 表示前 `i` 个房子能够获得的最大收益。

```cpp
class Solution {
public:
    int rob(vector<int>& nums) {
        vector<int> f(nums.size());
        if (nums.empty()) return 0;
        if (nums.size() == 1) return nums[0];
        f[0] = nums[0]; f[1] = max(nums[1], nums[0]);
        for (int i = 2; i < nums.size(); ++ i) {
            f[i] = max(f[i-2] + nums[i], f[i-1]);
        }
        return f.back();
    }
};
```

---

## 199. [二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/description/?envType=study-plan-v2&envId=top-100-liked)

如果我们每次都先递归搜索右子树，那么每个深度中第一个被访问的点就是被看到的点。

```cpp
class Solution {
    void dfs(TreeNode *root, int depth, vector<int> &ans) {
        if (root == nullptr) return ;
        if (depth == ans.size()) ans.push_back(root->val);
        dfs(root->right, depth+1, ans);
        dfs(root->left,  depth+1, ans);
    }
public:
    vector<int> rightSideView(TreeNode* root) {
        vector<int> ans;
        dfs(root, 0, ans);
        return ans;
    }
};
```

---

## 200. [岛屿数量](https://leetcode.cn/problems/number-of-islands/description/?envType=study-plan-v2&envId=top-100-liked)

循环遍历，遇到 1 的时候就通过 `dfs` 将所有相连的 1 变为 0。

```cpp
class Solution {
    void dfs(int u, int v, vector<vector<char>> &grid) {
        static int dx[] = {0, 0, 1, -1};
        static int dy[] = {1, -1, 0, 0};
        if (u < 0 || v < 0 || u >= grid.size() || v >= grid[0].size()) return ;
        if (grid[u][v] == '0') return ;
        grid[u][v] = '0';
        for (int k = 0; k < 4; ++ k) {
            int nx = u + dx[k];
            int ny = v + dy[k];
            dfs(nx, ny, grid);
        }
        return ;
    }
public:
    int numIslands(vector<vector<char>>& grid) {
        int ans = 0;
        for (int i = 0; i < grid.size(); ++ i) {
            for (int j = 0; j < grid[i].size(); ++ j) {
                if (grid[i][j] == '1') {
                    ++ ans;
                    dfs(i, j, grid);
                }
            }
        }
        return ans;
    }
};
```

---

## 206. [反转链表](https://leetcode.cn/problems/reverse-linked-list/description/?envType=study-plan-v2&envId=top-100-liked)

在数组中记录链表元素，然后逆序构造链表指针。

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if (head == nullptr) return nullptr;
        vector<ListNode*> sta;
        while(head != nullptr) {
            sta.push_back(head);
            head = head->next;
        }
        for (int i = sta.size()-1; i > 0; -- i) {
            sta[i]->next = sta[i-1];
        }
        sta[0]->next = nullptr;
        return sta.back();
    }
};
```

---

## 207. [课程表](https://leetcode.cn/problems/course-schedule/description/?envType=study-plan-v2&envId=top-100-liked)

拓扑排序

```cpp
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>>edges(numCourses);
        vector<int> deg;
        deg.resize(numCourses);
        for (auto vec: prerequisites) {
            int x = vec[0], y = vec[1];
            edges[x].push_back(y);
            ++ deg[y];
        }
        queue<int> q;
        for (int i = 0; i < numCourses; ++ i) {
            if (deg[i] == 0) q.push(i);
        }
        while(!q.empty()) {
            int u = q.front();q.pop();
            for (auto to: edges[u]) {
                if (-- deg[to] == 0) q.push(to);
            }
        }
        for (int i = 0; i < numCourses; ++ i) {
            if (deg[i] != 0) return false;
        }
        return true;
    }
};
```

---

## 208. [实现 Trie](https://leetcode.cn/problems/implement-trie-prefix-tree/description/?envType=study-plan-v2&envId=top-100-liked)

字面意思，实现一个字典树。写了一个动态的。

```cpp
struct Node {
    Node* ch[26];
    bool flag;
};
class Trie {
    Node *rt;
public:
    Trie() {
        rt = newNode();
    }

    Node* newNode() {
        Node *ret = new Node();
        for (int i = 0;i < 26; ++ i) {
            ret->ch[i] = NULL;
        }
        ret->flag = false;
        return ret;
    }

    void insert(string word) {
        Node *nw = rt;
        for (int i = 0;i < word.size(); ++ i) {
            if (nw->ch[word[i] - 'a'] == NULL) {
                nw->ch[word[i] - 'a'] = newNode();
            }
            nw = nw->ch[word[i] - 'a'];
        }
        nw->flag = true;
    }

    bool search(string word) {
        Node *nw = rt;
        for (int i = 0;i < word.size(); ++ i) {
            if (nw->ch[word[i] - 'a'] == NULL) return false;
            nw = nw->ch[word[i] - 'a'];
        }
        return nw->flag;
    }

    bool startsWith(string prefix) {
        Node *nw = rt;
        for (int i = 0;i < prefix.size(); ++ i) {
            if (nw->ch[prefix[i] - 'a'] == NULL) return false;
            nw = nw->ch[prefix[i] - 'a'];
        }
        return true;
    }
};
```

---

## 215. [数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2&envId=top-100-liked)

快速选择，用类似快拍的思想来做。每次找一个数字作为中间点，然后将所有小于等于该数字的放在左边，大于等于该数字的放在右边。然后找一个方向递归。

```cpp
class Solution {
    int dfs(vector<int>& nums, int l, int r, int k) {
        if (l == r) return nums[l];
        int mid = nums[l];
        int left = l - 1, right = r + 1;
        while(left < right) {
            do { ++ left; }while(nums[left] < mid);
            do { -- right; }while(nums[right] > mid);
            if (left < right) swap(nums[left], nums[right]);
        }
        if (r - right < k) {
            k -= (r - right);
            return dfs(nums, l, right, k);
        }else return dfs(nums, right+1, r, k);
    }
public:
    int findKthLargest(vector<int>& nums, int k) {
        return dfs(nums, 0, nums.size()-1, k);
    }
};
```

---

## 226. [翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked)

递归翻转左右子树。

```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) return nullptr;
        root->left = invertTree(root->left);
        root->right = invertTree(root->right);
        swap(root->left, root->right);
        return root;
    }
};
```

---

## 230. [二叉搜索树中第K小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/?envType=study-plan-v2&envId=top-100-liked)

用记忆化的方式计算节点的size，然后在二叉树上二分

```c++
class Solution {
public:
    unordered_map<TreeNode*, int> siz;
    int get_size(TreeNode *root) {
        if (root == NULL) return 0;
        if (siz.contains(root)) return siz[root];
        siz[root] = 1;
        if (root->left) siz[root] += get_size(root->left);
        if (root->right) siz[root] += get_size(root->right);
        return siz[root];
    }
    int kthSmallest(TreeNode* root, int k) {
        if (get_size(root->left) + 1 == k) {
            return root->val;
        }
        if (get_size(root->left) < k) {
            k -= get_size(root->left) + 1;
            return kthSmallest(root->right, k);
        } else return kthSmallest(root->left, k);
    }
};
```

---

## 234. [回文链表](https://leetcode.cn/problems/palindrome-linked-list/description/?envType=study-plan-v2&envId=top-100-liked)

进阶要求的话可以通过快慢指针找到中点，然后将后半部分翻转。但是这里就直接写了 copy 到数组中的版本。以及这个题其实可以用 SAM 做到 $O(1)$ 的时间复杂度。

```cpp
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        if (head == nullptr) return true;
        vector<int> vec;
        while(head != nullptr) {
            vec.push_back(head->val);
            head = head->next;
        }
        for (int i = 0, j = vec.size()-1; i < j; ++ i, -- j) {
            if (vec[i] != vec[j]) return false;
        }
        return true;
    }
};
```

---

## 235. [二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked)

两种方法，一种是记录 father 和深度，还可以倍增实现 $O(1)$ 查询，但是这个题只有一次查询，可以 $O(n)$ 暴力搜索。

```cpp
class Solution {
    bool dfs(TreeNode *root, TreeNode *p, TreeNode *q, TreeNode* &ans) {
        if (root == nullptr) return false;
        int l = dfs(root->left, p, q, ans);
        int r = dfs(root->right,p, q, ans);
        if ( (l && r) || ((root == p || root == q) && (l || r)) )
            ans = root;
        return l || r || (root == p) || (root == q);
    }
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        TreeNode *ans = nullptr;
        dfs(root, p, q, ans);
        return ans;
    }
};
```

---

## 238. [移动零](https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

双指针，一个指针遍历数组，另一个指针指向下一个非零元素应该存放的位置。

```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int j = 0;
        for (int i = 0; i < nums.size(); ++ i) {
            if (nums[i] == 0) continue;
            nums[j++] = nums[i];
        }
        while(j < nums.size()) nums[j++] = 0;
    }
};
```

---

## 238. [除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=top-100-liked)

两次遍历，第一次遍历计算左边的乘积，第二次遍历计算右边的乘积。

```cpp
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        vector<int> ret(nums.size());
        ret[0] = 1;
        for (int i = 1; i < nums.size(); ++ i){
            ret[i] = ret[i - 1] * nums[i-1];
        }
        int dotSum = nums.back();
        for (int i = nums.size() - 2; i >= 0; -- i) {
            ret[i] *= dotSum;
            dotSum *= nums[i];
        }
        return ret;
    }
};
```

---

## 239. [滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/?envType=study-plan-v2&envId=top-100-liked)

单调栈，维护递减的单调栈。

```rust
use std::collections::VecDeque;

impl Solution {
    pub fn max_sliding_window(nums: Vec<i32>, k: i32) -> Vec<i32> {
        let mut result: Vec<i32> = Vec::new();
        let mut sta: VecDeque<usize> = VecDeque::new();
        for i in 0..k-1 {
            while !sta.is_empty() && nums[i as usize] >= nums[*sta.back().unwrap()] {
                sta.pop_back();
            }
            sta.push_back(i as usize);
        }
        for i in k-1..nums.len() as i32{
            while !sta.is_empty() && nums[i as usize] >= nums[*sta.back().unwrap()] {
                sta.pop_back();
            }
            sta.push_back(i as usize);
            while i - *sta.front().unwrap() as i32 >= k {
                sta.pop_front();
            }
            result.push(nums[sta.front().unwrap().clone()])
        }
        result
    }
}
```

---

## 240. [搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/?envType=study-plan-v2&envId=top-100-liked)

从右上角开始搜索，如果 `target` 大于当前数字，则可以确定当前这一行都不会成为答案，可以向下走一步。如果 `target` 小于当前数字，则可以确定当前这一列都不会成为答案，可以向左走一步。时间复杂度 $O(n)$

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.size() == 0) return false;
        int x = 0, y = matrix[0].size() - 1;
        while(matrix[x][y] != target) {
            if (target > matrix[x][y]) ++ x;
            else if (target < matrix[x][y]) -- y;
            if (x >= matrix.size() || y < 0) return false;
        }
        return true;
    }
};
```

---

## 279. [完全平方数](https://leetcode.cn/problems/perfect-squares/description/?envType=study-plan-v2&envId=top-100-liked)

这个题动态规划，设`f[i]`表示和为 $i$ 的最少数量，然后枚举完全平方数 DP 。但是题解貌似用的是什么四平方数的数学定理，没太看懂）

```cpp
class Solution {
    int f[10010];
    void prework(int n) {
        f[0] = 0;
        for (int i = 1;i <= n; ++ i) {
            f[i] = i;
            for (int j = 1; j*j <= i; ++ j) {
                f[i] = min(f[i], f[i - j*j] + 1);
            }
        }
    }
public:
    Solution() {
        prework(10000);
    }
    int numSquares(int n) {
        return f[n];
    }
};
```

---

## 287. [寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)

很巧妙的思路，将数组中保存的数值看作连出的有向边，存在重复数字时一定存在环。快慢指针找到环之后，就可以将慢指针放到起点同步开始走，重合点就是入环点就是答案，可以列公式证明。

```cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int x = 0, y = 0;
        do {
            x = nums[nums[x]];
            y = nums[y];
        } while(x != y);
        y = 0;
        while(x != y) {
            x = nums[x];
            y = nums[y];
        }
        return x;
    }
};
```

---

## 300. [最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/?envType=study-plan-v2&envId=top-100-liked)

树状数组加速动态规划，动态保存权值的前缀区间内最大的数字是多少。

```cpp
#define lowbit(x) (x&(-x))
class Solution {
    int c[2510], n;
public:
    void modify(int x, int y) {
        for (;x<=n;x+=lowbit(x))
            c[x] = max(c[x],y);
    }
    int query(int x) {
        int ret = 0;
        for(;x;x-=lowbit(x))
            ret = max(ret, c[x]);
        return ret;
    }
    int lengthOfLIS(vector<int>& nums) {
        vector<int> sorted_nums = nums;
        sort(sorted_nums.begin(), sorted_nums.end());
        int ans = 0;
        n = sorted_nums.size() + 2;
        memset(c, 0, sizeof c);
        for (int i = 0; i < nums.size(); ++ i) {
            int x = lower_bound(sorted_nums.begin(), sorted_nums.end(), nums[i]) - sorted_nums.begin() + 2;
            int fx = query(x-1) + 1;
            ans = max(ans, fx);
            modify(x, fx);
        }
        return ans;
    }
};
```

---

## 322. [零钱兑换](https://leetcode.cn/problems/coin-change/description/?envType=study-plan-v2&envId=top-100-liked)

动态规划，`f[i]` 表示兑换 `i` 元的最少硬币数量。

```cpp
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<int> f(amount + 1, 0x3fffffff);
        f[0] = 0;
        for (auto coin: coins) {
            for (int i = coin; i <= amount; ++ i) {
                f[i] = min(f[i], f[i - coin] + 1);
            }
        }
        return f[amount] == 0x3fffffff ? -1 : f[amount];
    }
};
```

---

## 347. [前 K 个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/description/?envType=study-plan-v2&envId=top-100-liked)

用 HashMap 计算出每个数字出现的次数，然后用桶排序可以将复杂度控制到 $O(n)$

```cpp
class Solution {
    vector<int> ws[100010];
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> hash;
        for (int i = 0; i < nums.size(); ++ i) {
            if (hash.contains(nums[i])) {
                hash[nums[i]] += 1;
            } else {
                hash[nums[i]] = 1;
            }
        }
        for (int i = 0; i < nums.size(); ++i) ws[i].clear();
        for (auto it = hash.begin(); it != hash.end(); ++ it) {
            ws[it->second].push_back(it->first);
        }
        vector<int> ans;
        for (int i = nums.size(); i > 0; -- i) {
            if (ws[i].size() != 0) {
                for (int x : ws[i]) {
                    ans.push_back(x);
                    if (-- k == 0) break;
                }
            }
            if(k == 0) break;
        }
        return ans;
    }
};
```

---

## 394. [字符串解码](https://leetcode.cn/problems/decode-string/description/?envType=study-plan-v2&envId=top-100-liked)

丑陋！这段代码只能用丑陋来形容！

我怎么会写出如此丑陋的代码？括号序列+字符串处理真是难搞。

```cpp
class Solution {
public:
    string decodeString(string s) {
        string ans = "";
        stack<string> charStack;
        stack<int> numberStack;
        for (int i = 0; i < s.size(); ++ i) {
            if (s[i] == ']') {
                string ret = "";
                while(charStack.top() != "[") {
                    ret = charStack.top() + ret;
                    charStack.pop();
                }
                charStack.pop();
                for (int i = 0; i < numberStack.top(); ++ i) {
                    charStack.push(ret);
                }
                numberStack.pop();
            } else if (s[i] <= '9' && s[i] >= '0') {
                int x = s[i] - '0';
                while(s[i+1] <= '9' && s[i+1] >= '0') x = s[++i] - '0' + 10*x;
                numberStack.push(x);
            } else charStack.push(string(1, s[i]));
        }
        while(!charStack.empty()) {
            ans = charStack.top() + ans;
            charStack.pop();
        }
        return ans;
    }
};
```

---

## 416. [分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/description/?envType=study-plan-v2&envId=top-100-liked)

背包

```cpp
class Solution {
#define maxn 10000
    bool f[maxn + 1];
public:
    bool canPartition(vector<int>& nums) {
        int sum = 0;
        memset(f, 0, sizeof f);
        f[0] = true;
        for (auto x: nums) sum += x;
        if (sum % 2 != 0) return false;

        int lim = sum / 2;
        for (auto x: nums) {
            for (int i = lim; i >= x; -- i) {
                f[i] |= f[i - x];
            }
        }
        return f[lim];
    }
};
```

---

## 437. [路径总和 III](https://leetcode.cn/problems/path-sum-iii/description/?envType=study-plan-v2&envId=top-100-liked)

首先计算前缀和，用一个 HashMap 保存每一个数字是否出现，出现了多少次。然后通过搜索的方式在 HashMap 中查找以当前节点为端点的路径条数就可以了。

```cpp
class Solution {
    unordered_multiset<long long> hash;
    int ans;
    void dfs(TreeNode *u, long long sum, const int &targetSum) {
        if (u == nullptr) return ;
        sum += u->val;
        ans += hash.count(sum - targetSum);
        hash.insert(sum);
        dfs(u->left, sum, targetSum);
        dfs(u->right, sum, targetSum);
        hash.erase(hash.find(sum));
    }
public:
    int pathSum(TreeNode* root, int targetSum) {
        ans = 0;
        hash.insert(0);
        dfs(root, 0, targetSum);
        return ans;
    }
};
```

---

## 438. [找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked)

用桶来统计每一个字符出现的次数，然后滑动窗口不断维护窗口内的字符出现次数，维护匹配。

```cpp
class Solution {
    int wc[26], nw[26];
    void addChar(char c, int &matchCount) {
        int charIdx = c - 'a';
        nw[charIdx] += 1;
        if (nw[charIdx] == wc[charIdx]) ++ matchCount;
        else if(nw[charIdx] == wc[charIdx]+1) -- matchCount;
        return ;
    }
    void delChar(char c, int &matchCount) {
        int charIdx = c - 'a';
        nw[charIdx] -= 1;
        if (nw[charIdx] == wc[charIdx]) ++ matchCount;
        else if(nw[charIdx] == wc[charIdx]-1) -- matchCount;
        return ;
    }
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> ans;
        if(s.size() < p.size()) return ans;
        memset(wc, 0, sizeof wc);
        for (int i = 0; i < p.size(); ++ i){
            wc[p[i] - 'a'] += 1;
        }
        int matchCount = 0;
        for (int j = 0; j < 26; ++ j) {
            nw[j] = 0;
            if (nw[j] == wc[j]) ++ matchCount;
        }
        for (int i = 0; i < p.size(); ++ i) addChar(s[i], matchCount);
        if (matchCount == 26) ans.push_back(0);
        for (int i = p.size(); i < s.size(); ++ i) {
            addChar(s[i], matchCount);
            delChar(s[i-p.size()], matchCount);
            if (matchCount == 26) ans.push_back(i-p.size()+1);
        }
        return ans;
    }
};
```

---

## 543. [二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/description/?envType=study-plan-v2&envId=top-100-liked)

通过 dfs 计算出每一个点向下延伸的最大长度。

```cpp
class Solution {
    int dfs(TreeNode *root, int &ans) {
        if (root == nullptr) return -1;
        int left = dfs(root->left, ans) + 1;
        int right = dfs(root->right, ans) + 1;
        ans = max(ans, left + right);
        return max(left, right);
    }
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int ans = 0;
        dfs(root, ans);
        return ans;
    }
};
```

---

## 560. [和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/?envType=study-plan-v2&envId=top-100-liked)

数组和就是前缀和相减，用 hashmap 维护某个前缀数值出现的次数。

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> hash; hash.clear();
        int sum = 0, ans = 0;
        hash[sum] = 1;
        for (auto x: nums) {
            sum += x;
            ans += hash[sum - k];
            hash[sum] += 1;
        }
        return ans;
    }
};
```

---

## 739. [每日温度](https://leetcode.cn/problems/daily-temperatures/description/?envType=study-plan-v2&envId=top-100-liked)

倒序枚举，维护一个单调下降的单调栈，这样就可以快速对每个数字都快速找到之前出现的数字中第一个大于该数字的数字。

```cpp
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        stack<int> minStack;
        vector<int> ans(temperatures.size());
        for (int i = temperatures.size()-1; i >= 0; -- i) {
            while (!minStack.empty() &&
            temperatures[minStack.top()] <= temperatures[i]) {
                minStack.pop();
            }
            if (!minStack.empty()) ans[i] = minStack.top() - i;
            minStack.push(i);
        }
        return ans;
    }
};
```

---

## 763. [划分字母区间](https://leetcode.cn/problems/partition-labels/description/?envType=study-plan-v2&envId=top-100-liked)

计算出每个字母最后一次出现的位置，然后相当于区间覆盖问题了，从左到右依次枚举并维护右端点。

```cpp
class Solution {
public:
    vector<int> partitionLabels(string s) {
        static int last[26];
        memset(last, -1, sizeof last);
        vector<int> end(s.size());
        for (int i = s.size() - 1; i >= 0; -- i) {
            if (last[s[i] - 'a'] == -1) last[s[i] - 'a'] = i;
            end[i] = last[s[i] - 'a'];
        }
        vector<int> split;
        int ans = 0, r = -1, lastIdx = 0;
        for (int i = 0;i < s.size(); ++ i) {
            if (r < i) {
                ++ ans;
                int len = i - lastIdx;
                if (len != 0) split.push_back(len);
                lastIdx = i;
                r = end[i];
            } else r = max(r, end[i]);
        }
        split.push_back(s.size() - lastIdx);
        return split;
    }
};
```

---

## 994. [腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/description/?envType=study-plan-v2&envId=top-100-liked)

用队列保存待搜索的橘子位置，BFS。

```cpp
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        static int dx[] = {0, 0, 1, -1};
        static int dy[] = {1, -1, 0, 0};
        queue< pair<int,int> > q;
        int freshNumber = 0;
        for (int i = 0 ; i < grid.size(); ++ i) {
            for(int j = 0;j < grid[i].size(); ++ j) {
                if (grid[i][j] == 2) q.push(make_pair(i, j));
                else if(grid[i][j] == 1) ++ freshNumber;
            }
        }
        q.push(make_pair(-1, -1));
        int timeCounter = 0;
        while(!q.empty()) {
            int x = q.front().first;
            int y = q.front().second;
            q.pop();
            if (x == -1 && y == -1) {
                if (q.empty()) break;
                ++ timeCounter;
                q.push(make_pair(-1, -1));
                continue;
            }
            for (int k = 0; k < 4; ++ k) {
                int nx = x + dx[k];
                int ny = y + dy[k];
                if (nx < 0 || nx >= grid.size() || ny < 0 || ny >= grid[x].size())
                    continue;
                if (grid[nx][ny] != 1) continue;
                -- freshNumber;
                grid[nx][ny] = 2;
                q.push(make_pair(nx, ny));
            }
        }
        return freshNumber == 0 ? timeCounter : -1;
    }
};
```

---

## 1143. [最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/description/?envType=study-plan-v2&envId=top-100-liked)

$$
f[i][j] = \max\{f[i-1][j], f[i][j-1], [s_i == s_j](f[i-1][j-1]+1)\}
$$

```cpp
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        vector<vector<int>> f(text1.size()+1, vector<int>(text2.size()+1));
        for (int i = 1; i <= text1.size(); ++ i) {
            for (int j = 1; j <= text2.size(); ++ j) {
                if (text1[i-1] == text2[j-1]) f[i][j] = f[i-1][j-1] + 1;
                else f[i][j] = max(f[i-1][j], f[i][j-1]);
            }
        }
        return f[text1.size()][text2.size()];
    }
};
```
