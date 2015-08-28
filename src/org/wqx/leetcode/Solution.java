package org.wqx.leetcode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;

import javax.xml.stream.events.EndDocument;

class RandomListNode{
	int val;
	RandomListNode next;
	RandomListNode random;
	public RandomListNode(int val) {
		// TODO Auto-generated constructor stub
		this.val = val;
	}
}
class ListNode{
	int val;
	ListNode next;
	public ListNode(int val) {
		this.val = val;
	}
}
/**
 * Solution contains my solutions for <a href = "http://www.leetcode.com">leetcode</a>'s problems.
 * @author wanqiangxin
 *
 */
public class Solution {
	 public ListNode reverse(ListNode head){
		 ListNode pre = null, next = null;
		 while(head != null){
			 next = head.next;
			 head.next = pre;
			 pre = head;
			 head = next;			 
		 }
		 return pre;
	 }
	 public RandomListNode copyRandomList(RandomListNode head) {
	        HashMap<Integer, RandomListNode> nodeMap = new HashMap<Integer, RandomListNode>();
	        RandomListNode copyNode = new RandomListNode(0);
	        copyNode.next = null;
	        copyNode.random = null;
	        
	        RandomListNode cur = head;
	        RandomListNode curCopy = copyNode;
	        //1.In first pass, copy each node and it's next into copyNode.At the same time, put this node into hashmap with hashcode key;
	        while(cur!=null){
	           RandomListNode node = new RandomListNode(cur.val);
	           copyNode.next = node;
	           node.next = cur.next;
	           nodeMap.put(cur.hashCode(), cur);
	           cur = cur.next;
	        }
	        //2.In second pass, copy each node's random next
	        cur = head;
	         curCopy = copyNode.next;
	        while(cur!=null ){
	            curCopy.random = nodeMap.get(cur.next.hashCode());
	            curCopy = curCopy.next;
	            cur=cur.next;
	        }
	        
	        return copyNode.next;
	 }
	 
	 /**
	  * Given a non-negative number represented as an array of digits.<p>
	  * Plus one to the number.<p>
	  * The digits are stored such as the most significant digit is at the head of the array.<p>
	  * @param digits
	  * @return
	  */
	 public static int[] plusOne(int[] digits) {
	        int carry = 1;
	        for(int i = digits.length-1; i >=0; i--){
	        	int sum = (digits[i]+carry);
	            digits[i] =  sum % 10;
	            carry = sum / 10;
	        }
	        if (carry > 0){
	        	int [] re = new int [digits.length+1];
	        	System.arraycopy(digits, 0,re, 1, re.length-1);
	        	re[0] = carry;
	        	return re;
	        }
	        return digits;
	    }
	 /**
	  * Given a non-negative number represented as an array of digits.<p>
	  * Plus one to the number.<p>
	  * The digits are stored such as the most significant digit is at the head of the array.<p>
	  * @param digits
	  * @return
	  */
	 public static int[] plusOne2(int[] digits) {
		    if(digits == null || digits.length == 0){
		        return new int[0];
		    }

		    for(int i = digits.length - 1; i >= 0; i--){
		        if(digits[i] < 9){
		            digits[i]++;
		            return digits;
		        }else{
		            digits[i] = 0;
		        }
		    }

		    int[] result = new int[digits.length + 1];
		    result[0] = 1;

		    return result;
		}
	 
	    public boolean containsDuplicate(int[] nums) {
	        HashSet<Integer> numSet = new HashSet<Integer>();
	        for(int i =0; i<nums.length; i++){
	            if( numSet.contains(nums[i]) )return true;
	            else numSet.add(nums[i]);
	        }
	        return false;
	    }
	    /**
	     * Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that nums[i] = nums[j] and the difference between i and j is at most k.
	     * @param nums
	     * @param k
	     * @return
	     */
	    public boolean containsNearbyDuplicate(int[] nums, int k) {
	        HashMap<Integer, Integer> numMap = new HashMap<Integer, Integer>();
	        for(int i = 0; i< nums.length; i++){
	            if (numMap.containsKey(nums[i])){
	                if(i - numMap.get(nums[i]) > k) return true;
	                else numMap.put(nums[i], i);
	            }
	            numMap.put(nums[i],i);
	        }
	        return false;
	    }
	    public static int [] removeElement2(int[] nums, int val) {
	        int len = 0;
	        for(int i =0,j = i;j<nums.length;){
	            while(j<nums.length && nums[j] == val) j++;
	            if(j>=nums.length) break;
	            nums[i++] = nums[j++];
	            len++;
	            }
	        return nums;
	        }
	    /**
	     * Given an array and a value, remove all instances of that value in place and return the new length.<p>
	     * The order of elements can be changed. It doesn't matter what you leave beyond the new length.<p>
	     * @param nums
	     * @param val
	     * @return
	     */
	    public static int removeElement(int[] nums, int val) {
	    	int i = 0;
	        for(int j = 0; j < nums.length; j++ ){
	            if(nums[j] != val){
	                nums[i] = nums[j];
	                i++;
	            }
	        }
	        return i;
	    }
	  
	    public void rotate(int[] nums, int k) {
	        k = k % nums.length;
	        if (k == 0 ) return;
	        reverse(nums, 0, nums.length - 1);
	        System.out.println(Arrays.toString(nums));
	        reverse(nums, 0, k-1);
	        System.out.println(Arrays.toString(nums));
	        reverse(nums, k, nums.length - 1);
	        System.out.println(Arrays.toString(nums));
	        
	    }
	    public void reverse(int [] nums, int i, int j){
	        while(i++ < j --){
	            int tmp = nums[i];
	            nums[i] = nums[j];
	            nums[j] = tmp;
	        }
	    }
	    public int removeDuplicates(int[] nums) {
	        if(nums.length < 2) return nums.length;
	        int i = 1;
	        int val = nums[0];
	        for (int j = 1;j < nums.length; j++){
	            if(val != nums[j]){
	                nums[i] = nums[j];
	                val = nums[i];
	                i++;
	            }
	        }
	        System.out.println(Arrays.toString(nums));
	        return i;
	    }
	    /**
	     * Generate the Pascal's triangle
	     * @param numRows
	     * @return
	     */
	    public List<List<Integer>> generate(int numRows) {
	        List<List<Integer>> mtrix = new ArrayList<List<Integer>>();
	        int i = 0;
	        while(i<numRows){
	        	ArrayList<Integer> row = new ArrayList<Integer>();
	            int j = 0;
	            while(j<=i){
	                if(i == j){
	                	row.add(1);
	                }else if (i!=j && j == 0) {
	                	row.add(1);
					}else{
						row.add(mtrix.get(i-1).get(j-1)+mtrix.get(i-1).get(j));
					}
	                j++;
	            }
	            i++;
	            mtrix.add(row);
	        }
	        return mtrix;
	        
	    }
	    /**
	     * Generate the Pascal Triangle with last row only.<p>
	     * Hint: Compute the row from end to head.
	     * @param rowIndex
	     * @return
	     */
	    public List<Integer> getRow(int rowIndex) {
	        if(rowIndex==0) return null;
	        List<Integer> Kpascal = new ArrayList<Integer>(rowIndex);
	        for(int i = 0;i < rowIndex;i++){
	            Kpascal.add(0);
	        }
	        
	        Kpascal.set(0,1);
	        System.out.println(Kpascal);
	        for(int row = 1; row < rowIndex; row++){
	            for(int col = row; col > 0; col--){
	                Kpascal.set(col, Kpascal.get(col) + Kpascal.get(col-1));
	            }
	            System.out.println(Kpascal);
	        }
	        return Kpascal;
	    }
	    public void merge2(int[] nums1, int m, int[] nums2, int n) {
	        int i = 0, j = 0, k = m;
	        System.out.println(Arrays.toString(nums1));
	        System.out.println(Arrays.toString(nums2));
	        //merge nums1 and nums2 to nums1[m:(2m+n-1)]
	        while(i<m && j<n){
	            if(nums1[i]<nums2[j]){
	                nums1[k++] = nums1[i++];
	            }else{
	                nums1[k++] = nums2[j++];
	            }
	        }
	        System.out.println(Arrays.toString(nums1));
	        System.out.println(Arrays.toString(nums2));
	        while(i<m){
	            nums1[k++] = nums1[i++];
	        }
	        
	        while(j<n){
	            nums1[k++] = nums2[j++];
	        }
	        System.out.println(Arrays.toString(nums1));
	        System.out.println(Arrays.toString(nums2));
	        
	        //move nums1[m:(2m+n-1)] back to nums1[0:(m+n-1)]
	        
	        for(i = 0; i < m+n; i++){
	            nums1[i] = nums1[i+m];
	        }
	        System.out.println(Arrays.toString(nums1));
	        System.out.println(Arrays.toString(nums2));
	    }
	    /**
	     * Merge two sorted number list, nums1 and nums2, into nums1
	     * Hint: Merge them from end to head.
	     * @param nums1
	     * @param m
	     * @param nums2
	     * @param n
	     */
	    public void merge(int[] nums1, int m, int[] nums2, int n) {
	        int i = m-1, j = n-1, k = m+n-1;
	        while(i>=0 && j>=0)
	            nums1[k--] = nums1[i] >= nums2[j] ? nums1[i--]:nums2[j--];
	        while(j>=0)
	            nums1[k--] = nums2[j--];
	        
	        System.out.println(Arrays.toString(nums1));

	    }
	    public int threeSumClosest2(int[] nums, int target) {
	        int closed = Integer.MAX_VALUE;
	        int result = 0;
	        for(int i = 0; i< nums.length; i++)
	           for(int j = i+1; j< nums.length; j++)
	                for(int k = j+1; k< nums.length; k++){
	                	
	                    int n = Math.abs(target - (nums[i] + nums[j] +nums[k]));
	                   
	                    if( n < closed){
	                    	result = (nums[i] + nums[j] +nums[k]);
	                    	closed = n;
	                    	System.out.println(String.format("nums[%d]=%d,nums[%d]=%d,nums[%d]=%d,%d",i,nums[i],nums[j],j,k,nums[k],closed));
	                    }  
	                    
	                }
	        return result;
	                    
	    }
	    public int threeSumClosest(int[] nums, int target) {
	        int i = nums[0], j = nums[1], k = nums[2];
	        int closed = Math.abs(target - (i + j + k));
	        int result = i + j + k;
	        for(int n = 3; n < nums.length; n++){
                 int tmp = Math.abs(target - (nums[n] + j + k));
                    if(tmp < closed){
                        closed = tmp;
                        result = nums[n] + j + k;
                    }
                 tmp = Math.abs(target - (i + nums[n] + k));
                    if(tmp < closed){
                        closed = tmp;
                        result = i + nums[n] + k;
                    }
                 tmp = Math.abs(target - (i + j + nums[n]));
                    if(tmp < closed){
                        closed = tmp;
                        result = i + j + nums[n];
                    }
                
	        }
	        return result;
	    }
	    public int[] twoSum(int[] nums, int target) {
	        if (nums.length < 2) return null;
			 ArrayList<MyNumber> numslist = new ArrayList<MyNumber>();
			 int position = 0;
			 for(int n : nums){
				 numslist.add(new MyNumber(n, position));
				 position++;
			 }
			 
	        int [] result = {0,0};
	        Collections.sort(numslist, new Comparator<MyNumber>() {

				@Override
				public int compare(MyNumber o1, MyNumber o2) {
					
					return o1.compareTo(o2);
				}

	
			});
	        int first = 0;
	        int second = nums.length - 1;
	        while(first < second){
	            if(numslist.get(first).number + numslist.get(second).number == target){
	            	result[0] = numslist.get(first).position + 1;
	            	result[1] = numslist.get(second).position + 1;
	            	Arrays.sort(result);
	            	return result;
	            }
	            if(numslist.get(first).number + numslist.get(second).number < target) first++;
	            else second --;

	        }
	        return null;
	        
	    }
        class MyNumber implements Comparable<MyNumber>{
            int number;
            int position;
            public MyNumber(int number, int position){
                this.number = number;
                this.position = position;
            }

			@Override
			public int compareTo(MyNumber o) {
				if(this.number < o.number){
					return -1;
				}else if (this.number == o.number) {
					return 0;
				}else {
					return 1;
				}
			}


        }
        public int[] twoSum2(int[] nums, int target){
        	
        	
            HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
            int index = 0;
            for(int x : nums){
                if (map.containsKey(target - x)) return new int[]{map.get(target - x)+1, index+1};
                else map.put(x, index++);
            }
            return new int[2];
        }
	    public int threeSumClosest3(int[] nums, int target) {
	        if(nums.length < 3) return 0;
	        Arrays.sort(nums);
	        int closestSum = nums[0]+nums[1]+nums[nums.length - 1];
	        int diff = Math.abs(target - closestSum);
	        //for each i, find j and k, s.t.: nums[i] + nums[j] + nums[k] is closed to target.
	        for(int i = 0; i< nums.length; i++){
	            //As the nums is ascending order, let j = lowest, k = highest;
	            int j = 0, k = nums.length - 1; 
	            while(j < k){	           
    	            //if the sum is bigger than target, the highest shoud be smaller
    	            //otherwise, the lowest should be bigger.
    	            //each time, write down the closest sum.
    	            if(Math.abs(target - (nums[i] + nums[j] + nums[k])) < diff) {
    	                diff = Math.abs(target - (nums[i] + nums[j] + nums[k]));
    	                closestSum = nums[i] + nums[j] + nums[k];
    	            }
    	            if(nums[i] + nums[j] + nums[k] > target){
    	                k--;
    	            }else{
    	                j++;
    	            }
    	            if(j==i) j++;
    	            if(k==i) k--;
	            }
	        }
	        return closestSum;
	    }
	    public List<List<Integer>> threeSum(int[] nums) {
	        if(nums.length < 3) return null;
	        
	        Arrays.sort(nums); 
	        int len = nums.length;
	        List<List<Integer>> results = new ArrayList<List<Integer>>();
	        
	        for(int i = 0; i < len; i++){
	        	if(i == 0 || nums[i-1]==nums[i]) continue;
	            int j  = i + 1, k = len - 1;
	            while(j<k){
	                if(nums[i] + nums[j] + nums[k] == 0){
	                    ArrayList<Integer> result = new ArrayList<Integer>();
	                    result.add(nums[i]);
	                    result.add(nums[j]);
	                    result.add(nums[k]);
	                    results.add(result);
	                   
	                }
	                if(nums[i] + nums[j] + nums[k] > 0 ){
	                	k--;
	                    while(k>j & nums[k+1] == nums[k]) k--;
	                    
	                }else{
	                	j++;
	                	while(k>j& nums[j-1] == nums[j]) j++;	                 
	                }
	            }
	            
	        }
	        return results;
	    }
	    
	    public List<List<Integer>> threeSum2(int[] nums) {
       	    List<List<Integer>> results = new ArrayList<List<Integer>>();    
       	    if(nums.length < 3) return results;
       	    HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
       	    HashSet<List<Integer>> tracker = new HashSet<List<Integer>>();
       	    for(int i = 0; i< nums.length; i++){
       	        for(int j = i; j< nums.length; j++){
       	            if(map.containsKey(-nums[i]-nums[j])){
       	                ArrayList<Integer> result = new ArrayList<Integer>();
	                    int max = Math.max(-nums[i]-nums[j], Math.max(nums[i], nums[j]));
	                    int min = Math.min(-nums[i]-nums[j], Math.min(nums[i], nums[j]));
	                    result.add(min);
	                    result.add(-(max+min));
	                    result.add(max);
	                    if(tracker.add(result)){
	                    	results.add(result);
	                    }
	                    
       	            }else{
       	                map.put(nums[j],j);
       	            }
       	        }
       	     
       	    }
       	    return results;
	    }
	    public int maxArea(int[] height) {
	        if(height.length<2) return 0;
	        int max = 0, start = 0, end = 0;
	        
	        while(start < height.length && (end = nextMonotonyInterval(height, start)) < height.length){
	        	if (end == start) break;
	        	System.out.println(start+" "+end);
	        	int area = (height[start] + height[end])*(end - start)/2;
	        	max = max > area ? max : area;
	        	start = end ;
	        }
	        return max;
	    }
	    public int nextMonotonyInterval(int nums[], int start){
	    	int end = start + 1;
	    	while(end < nums.length){
	    		while(end < nums.length && nums[end-1] == nums[end]) end++;
	    		if(end < nums.length && nums[end-1] > nums[end]){
	    			while(end < nums.length && nums[end-1] > nums[end]) end++;
	    			while(end < nums.length && nums[end-1] == nums[end]) end++;
	    			return end - 1;
	    		}else if(end < nums.length && nums[end-1] < nums[end]){
	    			while(end < nums.length && nums[end-1] < nums[end]) end++;
	    			while(end < nums.length && nums[end-1] == nums[end]) end++;
	    			return end - 1;	    			
	    		}	
	    	}
	    	return start;
	    }
	    public int maxArea2(int[] height) {
	        int right = 0, left = height.length-1;
	        int max = 0;
	        while(right < left){
	            int area = Math.min(height[left], height[right]) * (left - right);
	            max = area > max ? area : max;
	            if(height[right] > height[left]) left --;
	            else right ++;
	        }
	        return max;
	        
	    }
	    public int findPeakElement2(int nums[]){
	    	int end = 1;
	    	while(end < nums.length){
	    		if(end < nums.length && nums[end-1] > nums[end]){
	    			while(end < nums.length && nums[end-1] > nums[end]) end++;
	    		}else if(end < nums.length && nums[end-1] < nums[end]){
	    			while(end < nums.length && nums[end-1] < nums[end]) end++;
	    			return end - 1;	    			
	    		}	
	    	}
	    	return 0;
	    }
	    public int findPeakElement3(int nums[]){
	        int left = 0, right = nums.length - 1;
	        while(left<right){
	            if(left + 1 == right){
	                return Math.max(nums[right], nums[left]);
	            }
	            int mid = left + (right - left) >> 1;
	            //mid - 1 and mid + 1 must be legal
	            if(nums[mid -1] < nums[mid] && nums[mid] > nums[mid + 1]) return mid;
	            else if(nums[mid -1] > nums[mid] && nums[mid] > nums[mid + 1]) left = mid + 1;
	            else right = mid - 1;
	        }
	        return 0;
	    }
	    public int findPeakElement(int nums[]){
	        int left = 0, right = nums.length - 1;
	        while(left<right){
	            int mid = left + (right - left)>>1;
	            //mid - 1 and mid + 1 must be legal
	            if(nums[mid] < nums[mid + 1]) left = mid + 1;
	            else right = mid;
	        }
	        return right;
	    }
	    public void sortColors(int[] nums) {
	    	int p = partition(nums, 0, 0);
	    	partition(nums, p, 1);
	        
	    }
	    public int partition(int []nums, int start, int color){
	        int low = start, high = nums.length - 1;
	        while(low < high){
	            while(low < high && nums[low] == color) low ++;
	            while(low < high && nums[high] != color) high --;
	            swap(nums, low, high);
	        }
	        return low;
	    }
	    public void swap(int[] nums, int i, int j){
	        int tmp = nums[i];
	        nums[i] = nums[j];
	        nums[j] = tmp;
	    }
	    public int findMin(int[] nums) {
	    	if(nums.length == 0) return -1;
	    	
	        int low = 0, high = nums.length - 1, mid = 0;
	        if(nums[low] < nums[high]) return nums[low];
	        while(low + 1 < high){
	            mid = low + (high - low)/2;
	            if(nums[mid] > nums[low]) low = mid;
	            else high = mid;
	            
	            
	        }
	        return nums[high];
	    }
	    public List<Integer> spiralOrder(int[][] matrix) {
	        List<Integer> results = new ArrayList<Integer>();
	        Set<Integer> set = new HashSet<Integer>();
	        set.addAll(results);
	        
	        if (matrix.length == 0 )return results;
	        helper(matrix, 0, 0, matrix.length-1, matrix[0].length - 1 , results);
	        return results;
	    }
	    // <i,j>-----------------------+
	    //   |  <i+1,j+1>              |
	    //   |                         |
	    //   |              <p-1,q-1>  |
	    //   +-----------------------<p,q>
	    public void helper(int[][] matrix, int i, int j, int p, int q,  List<Integer> results){
	    	while(true){
	    		//from top left to top right
	            for(int n = j; n <= q; n++) results.add(matrix[i][n]);
	            if(++i > p) break;
	            
	            //from top right to bottom right
	            for(int n = i ; n <= p; n++) results.add(matrix[n][q]);
	            if(--q < j) break;
	            
	            //from bottom right to bottom left
	            for(int n = q ;n >= j; n--) results.add(matrix[p][n]);
	            if(--p < i) break;
	            
	            //from bottom left to top left
	            for(int n = p ; n >= i; n--) results.add(matrix[n][j]);
	            if(++j > q) break;
	           
	            
	          

	    	}
	    	}

	        
	    public boolean canJump2(int[] nums) {
	        Set<Integer> set = new HashSet<Integer>();
	        int cur = 0;
	        while(set.add(cur)){
	            cur = (cur + nums[cur] ) % nums.length;
	            if(cur == nums.length - 1) return true;
	        }
	        return false;
	    }
	    public boolean canJump(int[] nums) {
	        int cur = 0;
	        int lastcur = 0;
	        while(cur<nums.length-1){
	        	lastcur = cur;
	            cur = (cur + nums[cur] );
	            if(cur == lastcur) return false;
	        }
	        return true;

	    }
	    public List<Integer> majorityElement(int[] nums) {
	        List<Integer> res = new ArrayList<Integer>();
	        if(nums.length == 0) {
	        	return res;
	        	}
	        int r1 = -123,r2 = -124, confidence1 = 0, confidence2= 0;	        
	        for (int n : nums) {
				if(confidence1 == 0 || n == r1){
					confidence1 ++;
					r1 = n;
				}else if(confidence2 == 0 || n == r2){
					confidence2 ++;
					r2 = n;
				}else{
					confidence1 --;
					confidence2 --;
				}
			}
	        
	        confidence1 = confidence2 = 0;
	        for(int n : nums){
	        	if(n==r1) confidence1++;
	        	if(n==r2) confidence2++;
	        }
	        if(confidence1 > nums.length /3) res.add(r1);
	        if(confidence2 > nums.length /3) res.add(r2);
		    return res;
	        
	    }
	    public int majorityElement2(int[] nums) {
	        int r = -1;
	        //confidence表示当前的待判定元素一共出现了几次
	        //1.若某元素和待判定元素相同，confidence++，若值大于n/2，则得到结果
	        //2.若某元素和待判定元素不同，则confidence--
	        //    2.1这时，若confidence==0，则说明待判定元素不是待求元素，需要再换
	        int confidence = 0;
	        int len = nums.length;
	        int half = len/3;
	        for(int i = 0;i<len;i++){
	            if(confidence==0){
	                r = nums[i];
	                confidence ++;
	            }else{
	                int b = nums[i]==r?(confidence++):(confidence--);
	                if (confidence > half ){return r;}
	                
	            }
	        }
			return r;
	    }  
	    public int maxSubArray(int[] nums) {
	    	
	        int maxsum = nums[0];
	        int maxsum_ending_with_j = -1;
	        for(int n : nums){
	            maxsum_ending_with_j = Math.max(maxsum_ending_with_j + n, n);
	            maxsum = Math.max(maxsum, maxsum_ending_with_j);
	        }
	        return maxsum;
	    }  
	    public int minPathSum1(int[][] grid) {
	   	        
	        return helper(grid, grid.length-1, grid[0].length - 1);
	    }
	    public int helper(int[][] grid, int i, int j){
	    	if(i == 0 && j == 0) return grid[i][j];
	    	else if(i == 0 && j != 0 ) return grid[i][j]+helper(grid, i, j-1);
	    	else if(j == 0 && i != 0) return grid[i][j]+helper(grid, i-1, j);
	    	else return grid[i][j]+Math.min(helper(grid, i-1, j), helper(grid, i, j-1));  	
	    }
	    //O(m*n) times ,O(m*n) sapce;
	    public int minPathSum2(int[][] grid) {
	    	int[][] shortestGrid = new int[grid.length][grid[0].length];
	    	shortestGrid[0][0] = grid[0][0];
	    	for(int j = 1; j < grid[0].length; j++) shortestGrid[0][j] = shortestGrid[0][j-1]+grid[0][j];
	    	for(int i = 1; i < grid.length; i++) shortestGrid[i][0] = shortestGrid[i-1][0]+grid[i][0];
	    	for(int i = 1; i < grid.length; i++){
	    		for(int j = 1; j< grid[0].length; j++){
	    			shortestGrid[i][j] = grid[i][j]+Math.min(shortestGrid[i-1][j], shortestGrid[i][j-1]);
	    		}
	    	}	    	
	    	return shortestGrid[shortestGrid.length -1 ][shortestGrid[0].length -1 ];	
	    }
	    
	  //O(m*n) times ,O(n) sapce;
	    public int minPathSum(int[][] grid) {
	    	int[] shortestGrid = new int[grid[0].length];
	    	shortestGrid[0]= grid[0][0];
	    	for(int j = 1; j < grid[0].length; j++) shortestGrid[j] = shortestGrid[j-1]+grid[0][j];
	    	
	    	for(int i = 1; i < grid.length; i++){
	    		for(int j = 0; j< grid[0].length; j++){
	    			if( j == 0)shortestGrid[j] = grid[i][0] + shortestGrid[j];
	    			else shortestGrid[j] = grid[i][j]+Math.min(shortestGrid[j-1], shortestGrid[j]);
	    		}	
	    	}
	    	
	    	return shortestGrid[grid[0].length - 1];
	    }
	    public int climbStairs(int n) {
	        if(n==0) return 0;
	        if(n==1) return 1;
	        if(n==2) return 1;
	        int flow1 = 1;
	        int flow2 = 1;
	        int flow3 = 0;
	        int i = 3;
	        while(i<=n){
	           flow3 = flow2 + flow1;
	           flow1 = flow2;
	           flow2 = flow3;
	           i++;
	           System.out.println(flow3+" ");
	        }
	        return flow3;
	        
	    }
	    public int minimumTotal(List<List<Integer>> triangle) {
	        int []sps = new int[triangle.size() + 1];
	        for (int i = 0; i < sps.length; i++) {
				sps[i] = Integer.MAX_VALUE;
			}
	        
	        sps[1] = triangle.get(0).get(0);
	        int min = sps[1];
	        for(int i = 1; i < triangle.size(); i++){
	        	min = sps[1];
	        	for(int j =  triangle.get(i).size() - 1; j >= 0; j--){
	        		sps[j+1] = Math.min(sps[j+1], sps[j])+triangle.get(i).get(j);
	        		min = Math.min(sps[j+1], min);
	        	}
	        }
	        return min;
	        
	    } 
	    public int minimumTotal2(Integer[][] triangle) {
	        int []sps = new int[triangle.length + 1];
	        for (int i = 0; i < sps.length; i++) {
				sps[i] = Integer.MAX_VALUE;
			}
	        
	        sps[1] = triangle[0][0];
	        int min = sps[1];
	       
	        for(int i = 1; i < triangle.length; i++){
	        	min = Integer.MAX_VALUE;
	        	for(int j = triangle[i].length - 1; j >=0 ; j--){
	        		sps[j+1] = Math.min(sps[j+1], sps[j])+triangle[i][j];
	        		min= Math.min(sps[j+1], min);
	        	}
	        	printArray(sps);
	        }
	        return min;
	        
	    } 
	    public int uniquePaths(int m, int n) {
	    	if(m*n ==0) return 0;
	    	int []uiques = new int[n+1];
	    	uiques[0] = 0;
	    	for (int i = 1; i < uiques.length; i++) {
				uiques[i] = 1;
			}
	    	printArray(uiques);
	    	for (int i = 1; i < m; i++) {
	    		
				for (int j = 1; j < uiques.length; j++) {
					uiques[j] = uiques[j] + uiques[j-1];
				}
				printArray(uiques);
			}
	    	return uiques[n];
	    }
	    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
	        if(obstacleGrid.length == 0) return 0;
	        int []uiques = new int[obstacleGrid[0].length+1];
	        uiques[0] = 0;
	        
	    	for (int i = 1; i < uiques.length; i++) {
	    		if(obstacleGrid[0][i-1] == 1) break;
	    		uiques[i] = 1;
			}
	    	printArray(uiques);
	    	for (int i = 1; i < obstacleGrid.length; i++) {
	    		
				for (int j = 1; j < uiques.length; j++) {
					uiques[j] = obstacleGrid[i][j-1] == 1 ? 0 :uiques[j] + uiques[j-1];
				}
				printArray(uiques);
			}
	    	return uiques[obstacleGrid[0].length];
	        
	        
	    }
	    public int maxProduct(int[] nums) {
	    	if(nums.length == 1) return nums[0];
	        int maxProduct = nums[0];
	        int minProduct = nums[0];
	        int maxProduct_end_with_j = maxProduct;
	        int minProduct_end_with_j = minProduct;
	        for (int j = 1; j < nums.length; j++) {
	        	int tmp = maxProduct_end_with_j;
				maxProduct_end_with_j = Math.max(maxProduct_end_with_j*nums[j], minProduct_end_with_j*nums[j]);
				minProduct_end_with_j = Math.min(tmp*nums[j], minProduct_end_with_j*nums[j]);
				maxProduct = Math.max(maxProduct, maxProduct_end_with_j);
				maxProduct = Math.max(maxProduct, nums[j]);
			}
	        
	        return maxProduct;
	    }
	    /**
	     * Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
	     * @param n
	     * @return
	     */
	    public int numTrees(int n) {
	    	//Catalan number Problem
	        //if n == 1 return 1
	        if(n <= 1) return 1;
	        int[] s = new int[n+1];
	        s[0]=s[1]=1;
	        for(int i = 2; i < n+1; i++){
	            s[i]=0;
	            for(int j = 0; j < i; j++){
	                s[i] = s[i] + s[j] * s[i-j-1];
	            }
	        }
	        return s[n];
	        
	    }
	 public static void printArray(int []array){
		 System.out.println(Arrays.toString(array));
	 }
	 public static void main(String[] args) {
		 int[] a = {1};
		 
		 
		 Solution s= new Solution();
////		 s.rotate(a, 2);
//		 //System.out.println(s.removeDuplicates(a));
//		 for(List<Integer> row : s.generate(10))
//			 System.out.println(row);
//		 ArrayList<Integer> row = new ArrayList<Integer>(10);
//		 System.out.println(s.getRow(10));
		 int [] nums1 = {-1,2,1,-4};
//		System.out.println(s.threeSumClosest(nums1, 82));
//		Arrays.sort(nums1);

//		System.out.println(s.threeSumClosest3(nums1, 1));
//System.out.println(s.threeSum(new int[]{7,-1,14,-12,-8,7,2,-15,8,8,-8,-14,-4,-5,7,9,11,-4,-15,-6,1,-14,4,3,10,-5,2,1,6,11,2,-2,-5,-7,-6,2,-15,11,-6,8,-4,2,1,-1,4,-6,-15,1,5,-15,10,14,9,-8,-6,4,-6,11,12,-15,7,-1,-9,9,-1,0,-4,-1,-12,-2,14,-9,7,0,-3,-4,1,-2,12,14,-10,0,5,14,-1,14,3,8,10,-8,8,-5,-2,6,-11,12,13,-7,-12,8,6,-13,14,-2,-5,-11,1,3,-6}));
//		 System.out.println(s.findPeakElement(new int[]{1,2,3,4}));
		 int [] nums = {1,2};
//		 s.sortColors(nums);
//		System.out.println(s.partition(nums, 0, 0));
//		printArray(nums);
//		System.out.println(s.partition(nums, 4, 1));
//		printArray(nums);
//		
//		System.out.println(s.findMin(new int[]{5,4,3,2,1,0,-1,-2})); 
//		 System.out.println(s.spiralOrder(new int[][]{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}));
//		 System.out.println(s.spiralOrder(new int[][]{{1,2},{4,5},{7,8},{10,11}}));
//		 System.out.println(s.spiralOrder(new int[][]{{5},{8}}));
//		 System.out.println(s.spiralOrder(new int[][]{{5},{8},{9}}));
//		 System.out.println(s.spiralOrder(new int[][]{{1,2,3}}));
//		 System.out.println(s.spiralOrder(new int[][]{{1,2,3,4},{1,2,3,4},{1,2,3,4},{1,2,3,4}}));
//		 System.out.println(s.spiralOrder(new int[][]{{2,5,8},{4,0,9}}));
////		 System.out.println(s.canJump(new int[]{1,0,1}));
//		 System.out.println(s.majorityElement(new int[]{}));
//		 System.out.println(s.majorityElement(new int[]{1}));
//		 System.out.println(s.majorityElement(new int[]{1,2}));
//		 System.out.println(s.majorityElement(new int[]{1,1}));
//		 System.out.println(s.majorityElement(new int[]{1,2,3}));
//		 System.out.println(s.majorityElement(new int[]{1,1,2,5}));
//		 System.out.println(s.majorityElement(new int[]{1,1,1,1,2,2,2,2,3,3}));
//		 System.out.println(s.majorityElement(new int[]{3,2,3}));
//		 System.out.println((5<5?10.0:9));
//		 System.out.println(s.maxSubArray(new int[]{-2,11,-4,13,-5,-2}));
//		 System.out.println(s.maxSubArray(new int[]{-2,-4,-5,-2}));
//		 System.out.println(s.maxSubArray(new int[]{-2,-4,-5,-2}));
//		 System.out.println(s.maxSubArray(new int[]{-2}));
//		 System.out.println(s.minPathSum(new int[][]{{1,2,3},{4,5,6},{7,8,9}}));
//		 System.out.println(s.minPathSum(new int[][]{{1,2,3},{4,1,1},{7,8,9}}));
//		 System.out.println(s.minPathSum(new int[][]{{1},{1}}));
//		 System.out.println(s.minPathSum(new int[][]{{5,0,1,1,2,1,0,1,3,6,3,0,7,3,3,3,1},{1,4,1,8,5,5,5,6,8,7,0,4,3,9,9,6,0},{2,8,3,3,1,6,1,4,9,0,9,2,3,3,3,8,4},{3,5,1,9,3,0,8,3,4,3,4,6,9,6,8,9,9},{3,0,7,4,6,6,4,6,8,8,9,3,8,3,9,3,4},{8,8,6,8,3,3,1,7,9,3,3,9,2,4,3,5,1},{7,1,0,4,7,8,4,6,4,2,1,3,7,8,3,5,4},{3,0,9,6,7,8,9,2,0,4,6,3,9,7,2,0,7},{8,0,8,2,6,4,4,0,9,3,8,4,0,4,7,0,4},{3,7,4,5,9,4,9,7,9,8,7,4,0,4,2,0,4},{5,9,0,1,9,1,5,9,5,5,3,4,6,9,8,5,6},{5,7,2,4,4,4,2,1,8,4,8,0,5,4,7,4,7},{9,5,8,6,4,4,3,9,8,1,1,8,7,7,3,6,9},{7,2,3,1,6,3,6,6,6,3,2,3,9,9,4,4,8}}));
//		 
//		 System.out.println(s.minPathSum2(new int[][]{{1,2,3},{4,5,6},{7,8,9}}));
//		 System.out.println(s.minPathSum2(new int[][]{{1,2,3},{4,1,1},{7,8,9}}));
//		 System.out.println(s.minPathSum2(new int[][]{{1},{1}}));
//		 System.out.println(s.minPathSum2(new int[][]{{5,0,1,1,2,1,0,1,3,6,3,0,7,3,3,3,1},{1,4,1,8,5,5,5,6,8,7,0,4,3,9,9,6,0},{2,8,3,3,1,6,1,4,9,0,9,2,3,3,3,8,4},{3,5,1,9,3,0,8,3,4,3,4,6,9,6,8,9,9},{3,0,7,4,6,6,4,6,8,8,9,3,8,3,9,3,4},{8,8,6,8,3,3,1,7,9,3,3,9,2,4,3,5,1},{7,1,0,4,7,8,4,6,4,2,1,3,7,8,3,5,4},{3,0,9,6,7,8,9,2,0,4,6,3,9,7,2,0,7},{8,0,8,2,6,4,4,0,9,3,8,4,0,4,7,0,4},{3,7,4,5,9,4,9,7,9,8,7,4,0,4,2,0,4},{5,9,0,1,9,1,5,9,5,5,3,4,6,9,8,5,6},{5,7,2,4,4,4,2,1,8,4,8,0,5,4,7,4,7},{9,5,8,6,4,4,3,9,8,1,1,8,7,7,3,6,9},{7,2,3,1,6,3,6,6,6,3,2,3,9,9,4,4,8}}));
//		 
//		 s.climbStairs(10);
//		System.out.println(s.minimumTotal2(new Integer[][]{{2},{3,4},{6,5,7},{4,1,8,3}}));
//		System.out.println(s.minimumTotal2(new Integer[][]{{1},{2,3}}));
//		System.out.println(s.uniquePaths(3, 7));
		//System.out.println(s.uniquePathsWithObstacles(new int[][]{{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,0,0}}));
//		System.out.println(s.uniquePathsWithObstacles(new int[][]{{0,0,1,0},{0,0,0,0},{0,0,0,0}}));
//		System.out.println(s.uniquePathsWithObstacles(new int[][]{{1,0}}));
//		System.out.println(s.maxProduct(new int[]{2,3,-2,4}));
//		System.out.println(s.maxProduct(new int[]{-4,4}));
//		System.out.println(s.maxProduct(new int[]{2,-2,-2,4}));
//		System.out.println(s.maxProduct(new int[]{0,2,3,-2,4}));
//		System.out.println(s.maxProduct(new int[]{-2,3,-4}));
		System.out.println(s.numTrees(0));
		System.out.println(s.numTrees(1));
		System.out.println(s.numTrees(2));
		System.out.println(s.numTrees(3));
		System.out.println(s.numTrees(4));
	}

}
